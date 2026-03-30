"""Speech Encoder using HuBERT-large"""

import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel, Wav2Vec2Processor
from typing import Union, List, Optional
import numpy as np
import os


class SpeechEncoder(nn.Module):
    """
    Speech encoder wrapper for HuBERT-large.
    Processes raw audio at 16kHz to generate deep speech representations.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ls960-ft",
        freeze: bool = True,
        target_sample_rate: int = 16000,
        token: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze model parameters
            target_sample_rate: Target audio sample rate (16kHz)
            token: HuggingFace token for private models (defaults to HF_TOKEN env var)
        """
        super().__init__()
        self.model_name = model_name
        self.target_sample_rate = target_sample_rate
        
        # Get token from parameter, environment variable, or None
        hf_token = token or os.getenv("HF_TOKEN")
        
        # Load HuBERT model and processor
        # Use token only if provided (for private models)
        load_kwargs = {}
        if hf_token:
            load_kwargs["token"] = hf_token
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, **load_kwargs)
        self.model = HubertModel.from_pretrained(model_name, **load_kwargs)
        
        # Freeze all parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Get output dimension
        self.hidden_size = self.model.config.hidden_size  # 1024 for HuBERT-large
        print(f"Initialized SpeechEncoder with model '{model_name}' (hidden size: {self.hidden_size}, freeze: {freeze})")

        self.dummy_input = torch.zeros(1, self.target_sample_rate)  # 1 second of silence for dummy input
        self.dummy_input = self.dummy_input.to(next(self.model.parameters()).device)
        tmp_audio = self.preprocess_audio(self.dummy_input)
        print(f"Dummy audio shape after preprocessing: {tmp_audio.shape}")
        tmp_inputs = self.encode(tmp_audio)

    
    def preprocess_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray, str],
        sample_rate: int = None
    ) -> torch.Tensor:
        """
        Preprocess audio to 16kHz mono format.
        
        Args:
            audio: Audio tensor/array or path to audio file
            sample_rate: Original sample rate (if audio is tensor/array)
        
        Returns:
            Preprocessed audio tensor at 16kHz
        """
        # Load audio if path provided
        # Padronize input to (channels, samples) format
        if isinstance(audio, str):

            waveform, sample_rate = torchaudio.load(audio)

        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()

        elif isinstance(audio, torch.Tensor):
            waveform = audio

        else:
            raise ValueError("Audio input must be a file path, numpy array, or torch tensor.")

        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # (1, samples)

        elif waveform.ndim == 2 and waveform.shape[0] > 1:
            # Convert multi-channel audio to mono by averaging channels.
            waveform = waveform.mean(dim=0, keepdim=True)

        elif waveform.ndim != 2:
            raise ValueError(f"Unexpected audio tensor shape: {tuple(waveform.shape)}")
        
        assert waveform.shape[0] == 1, "Audio should be mono (1 channel) after preprocessing."

        if sample_rate is not None and sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)   # Return as (1, samples) tensor
    
    def encode(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, List],
        normalize: bool = False,
        device: str = None
    ) -> torch.Tensor:
        """
        Encode audio into speech representations.
        
        Args:
            audio: Audio input(s) - tensor, array, file path, or list
            normalize: Whether to normalize output (usually not needed for adapter input)
            device: Device to run on (auto-detect if None)
        
        Returns:
            Hidden states tensor of shape (batch_size, seq_len, hidden_size)
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        # Handle batch input
        if isinstance(audio, list):
            # Preprocess each audio
            waveforms = []
            for a in audio:
                wav = self.preprocess_audio(a)
                waveforms.append(wav)

        elif isinstance(audio, (torch.Tensor, np.ndarray)):

            if audio.ndim == 1 or (audio.ndim == 2 and audio.shape[0] == 1):
                waveforms = [self.preprocess_audio(audio).to(device)]
            else:
                # Pass [batch, samples] - each item have to be preprocessed separately
                waveforms = []
                for i in range(audio.shape[0]):
                    wav = self.preprocess_audio(audio[i])
                    waveforms.append(wav)
        # print(f"Preprocessed {len(waveforms)} audio samples for encoding.")
        # print(f"Example preprocessed audio shape: {waveforms[0].shape}")

        hidden_states_list = []
        for idx, wav in enumerate(waveforms):
            inputs = self.processor(
                wav,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                padding=True
            ).to(device)

            # print(f"Audio {idx} processor output keys: {inputs}")
            
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.model(**inputs)
                hidden_states_list.append(outputs.last_hidden_state)

        hidden_states = torch.cat(hidden_states_list, dim=0)  # (batch_size, seq_len, hidden_size)
        # print(f"Encoded hidden states shape: {hidden_states.shape}")
        
        if normalize:
            hidden_states = nn.functional.normalize(hidden_states, p=2, dim=-1)
        
        return hidden_states
    
    def forward(self, audio: Union[torch.Tensor, np.ndarray, str, List], **kwargs) -> torch.Tensor:
        """Forward pass for compatibility with nn.Module"""
        return self.encode(audio, **kwargs)

