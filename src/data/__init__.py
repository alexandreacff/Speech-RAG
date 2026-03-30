"""Data loading and preprocessing."""

from .dataset import RetrievableSpeechDataset, speech_collate_fn
from .preprocessing import AudioPreprocessor

# Backward-compatible alias used by scripts/train.py
SpeechDataset = RetrievableSpeechDataset

__all__ = ["RetrievableSpeechDataset", "SpeechDataset", "AudioPreprocessor", "speech_collate_fn"]

