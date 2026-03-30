"""Audio preprocessing utilities"""

import torch
import torchaudio
import numpy as np
from typing import Union, Tuple
import librosa

class AudioPreprocessor:
    """Audio preprocessing for speech models"""
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        max_length_seconds: float = 60.0, # Increased to 60s for Spoken-SQuAD
        normalize: bool = False
    ):
        self.target_sample_rate = target_sample_rate
        self.max_length_seconds = max_length_seconds
        self.max_length_samples = int(max_length_seconds * target_sample_rate)
        self.normalize = normalize
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file safely."""

        waveform, sr = torchaudio.load(audio_path) 
        
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)  # (samples,) -> (1, samples)
            
        return waveform, sr
    
    def resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if sample_rate == self.target_sample_rate:
            return waveform
            
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=self.target_sample_rate
        )
        return resampler(waveform)

    def trim_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trim or pad audio to fixed length."""
            
        current_len = waveform.shape[-1]
        target_len = self.max_length_samples
        
        if current_len > target_len:
            waveform = waveform[..., :target_len]
        elif current_len < target_len:
            pad_amt = target_len - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
            
        return waveform
    
    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert stereo to mono by averaging channels."""
        if waveform.dim() == 1:          # já mono: [T]
            return waveform.unsqueeze(0) # padroniza para [1, T]
        if waveform.dim() == 2 and waveform.shape[0] > 1:  # [C, T], C>1
            return waveform.mean(dim=0, keepdim=True)      # -> [1, T]
        return waveform  # já [1, T]

    def process(self, audio_input: Union[str, torch.Tensor]) -> torch.Tensor:
        """Complete pipeline."""
        # 1. Load
        if isinstance(audio_input, str):
            waveform, sr = self.load_audio(audio_input)
        else:
            waveform = audio_input
            sr = self.target_sample_rate # Assume correct if tensor passed directly

        # 1. Convert to mono
        waveform = self.to_mono(waveform)

        # 2. Resample
        waveform = self.resample(waveform, sr)
        
        # 3. Trim/Pad
        # waveform = self.trim_or_pad(waveform)
        
        # 4. Normalize
        if self.normalize:
            if torch.abs(waveform).max() > 0:
                waveform = waveform / torch.abs(waveform).max()
                
        return waveform.squeeze(0)  # Return (samples,)