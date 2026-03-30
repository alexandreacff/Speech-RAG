"""
Data Check Script for RetrievableSpeechDataset
"""

from pathlib import Path
import sys
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import RetrievableSpeechDataset, speech_collate_fn
from torch.utils.data import DataLoader


def main():
    # Paths (adjust as needed)
    metadata_path = Path("../../speech-rag/src/data/spoken_train-v1.1.json")
    audio_dir = Path("../../speech-rag/src/data/train_wav/")

    # Initialize dataset
    dataset = RetrievableSpeechDataset(
        metadata_path=str(metadata_path),
        audio_dir=str(audio_dir),
        sample_rate=16000,
        max_audio_length=120.0
    )

    # Print summary
    print(f"Total samples loaded: {len(dataset.samples)}")
    if dataset.samples:
        print(f"Example sample: {dataset.samples[0]}")

    # Use dataloader to check batching
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=speech_collate_fn)

    for batch in dataloader:
        print("Batch audio shape:", batch["audio"].shape)
        print("Batch attention mask shape:", batch["attention_mask"].shape)
        print("Batch text samples:", batch["text"])
        break  # Just check one batch

if __name__ == "__main__":
    main()
    