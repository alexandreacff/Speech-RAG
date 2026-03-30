
import json
import os
import torch
import soundfile as sf
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path
from .preprocessing import AudioPreprocessor
from glob import glob
from tqdm import tqdm

import mutagen

class RetrievableSpeechDataset(Dataset):
    def __init__(
        self, 
        metadata_path: str, 
        audio_dir: str, 
        sample_rate: int = 16000,
        max_audio_length: float = 60.0
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=sample_rate,
            max_length_seconds=max_audio_length
        )
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            raw_data = json.load(f)

        self.samples = []
        missing_count = 0
        
        # --- NEW LOGIC: Construct filenames based on INDEX ---
        # Format: {article_idx}_{paragraph_idx}_{qa_idx}.wav
        
        for article_idx, article in enumerate(raw_data['data']):
            for para_idx, paragraph in enumerate(article['paragraphs']):
                context_text = paragraph['context']
                sentences = [s.strip() for s in context_text.split('.') if s.strip()]
            
                for sentence_idx, sentence in enumerate(sentences):
                    
                    # Construct the filename expected by Spoken-SQuAD structure
                    filename = f"{article_idx}_{para_idx}_{sentence_idx}.wav"

                    try:
                        audio_duration = mutagen.File(self.audio_dir / filename).info.length
                        # print(f"Checking {filename}: duration {audio_duration:.2f}s")
                        if audio_duration > self.max_audio_length:
                            print(f"Skipping {filename} due to length ({audio_duration:.2f}s > {self.max_audio_length}s)")
                            missing_count += 1
                            continue

                    except Exception as e:
                        print(f"Error checking {filename}: {e}")
                        missing_count += 1
                        continue

                    audio_path = self.audio_dir / filename
                    
                    if audio_path.exists():
                        self.samples.append({
                            "id": f"{article_idx}_{para_idx}_{sentence_idx}", # Use our generated ID
                            "audio_path": str(audio_path),
                            "full_context": context_text,
                            "sentence": sentence,
                            "question": ""
                        })
                    else:
                        missing_count += 1

        print(f"Matched {len(self.samples)} samples. (Missing/Skipped: {missing_count})")
        
        if len(self.samples) == 0:
             # Fallback debug: print what we TRIED to look for
            print(f"DEBUG: First expected file was {self.audio_dir}/0_0_0.wav")
            raise ValueError(f"No valid samples found in {audio_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        waveform = self.preprocessor.process(item["audio_path"])
        
        return {
            "audio": waveform,
            "text": item["sentence"]
        }

def speech_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # Filter failed loads
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    audios = [item["audio"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # Pad Audios
    lengths = torch.tensor([audio.shape[0] for audio in audios])
    max_len = lengths.max().item()
    
    padded_audios = torch.zeros(len(audios), max_len)
    attention_masks = torch.zeros(len(audios), max_len)
    
    for i, audio in enumerate(audios):
        end = audio.shape[0]
        padded_audios[i, :end] = audio
        attention_masks[i, :end] = 1
        
    return {
        "audio": padded_audios,
        "attention_mask": attention_masks,
        "text": texts
    }