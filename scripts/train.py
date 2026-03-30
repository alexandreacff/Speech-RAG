"""Training script for Speech Retriever"""

import argparse
import math
import yaml
import torch
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
# Added speech_collate_fn import
from src.data import SpeechDataset, speech_collate_fn
from training.trainer import Trainer
from training.losses import DistillationLoss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train Speech Retriever")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Fallback for Mac M1/M2 if cuda is not available
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    finetune_text_encoder = bool(config["training"].get("finetune_text_encoder", False))
    finetune_speech_encoder = bool(config["training"].get("finetune_speech_encoder", True))

    text_encoder = TextEncoder(
        model_name=config["models"]["text_encoder"],
        freeze=not finetune_text_encoder
    )
    
    speech_encoder = SpeechEncoder(
        model_name=config["models"]["speech_encoder"],
        freeze=not finetune_speech_encoder
    )
    
    # Get embedding dimensions
    text_embedding_dim = text_encoder.embedding_dim
    speech_hidden_dim = speech_encoder.hidden_size
    
    # Create adapter
    adapter = SpeechAdapter(
        input_dim=speech_hidden_dim,
        output_dim=text_embedding_dim,
        downsample_factor=4
    )
    
    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")
    
    # --- LOAD DATASETS (UPDATED) ---
    print("Loading datasets...")
    
    # Get data root from config, default to local 'data' folder
    data_root = config["paths"].get("data_dir", "data")
    
    # 1. Training Set
    # Maps to: data/spoken_train-v1.1.json AND data/train_wav
    train_metadata = os.path.join(data_root, "spoken_train-v1.1.json")
    train_audio_dir = os.path.join(data_root, "train_wav")
    
    if not os.path.exists(train_metadata) or not os.path.exists(train_audio_dir):
        raise FileNotFoundError(
            f"Training data not found at {train_metadata} or {train_audio_dir}"
        )

    train_dataset = SpeechDataset(
        metadata_path=train_metadata,
        audio_dir=train_audio_dir,
        sample_rate=config["data"]["sample_rate"],
        max_audio_length=config["data"]["max_audio_length"]
    )
    print(f"Training dataset size: {len(train_dataset)}")

    # 2. Validation Set
    # Maps to: data/spoken_test-v1.1.json AND data/dev_wav
    val_dataset = None
    val_metadata = os.path.join(data_root, "spoken_test-v1.1.json")
    val_audio_dir = os.path.join(data_root, "dev_wav")

    if os.path.exists(val_metadata) and os.path.exists(val_audio_dir):
        try:
            val_dataset = SpeechDataset(
                metadata_path=val_metadata,
                audio_dir=val_audio_dir,
                sample_rate=config["data"]["sample_rate"],
                max_audio_length=config["data"]["max_audio_length"]
            )
            print(f"Validation dataset size: {len(val_dataset)}")
        except Exception as e:
            print(f"Could not load validation dataset: {e}")
    else:
        print(f"Warning: Validation data not found at {val_audio_dir}. Skipping validation.")

    # Loss function
    loss_fn = DistillationLoss(
        loss_type=config["training"]["loss_type"]
    )
    
    # Optimizer
    learning_rate = float(config["training"]["learning_rate"])
    speech_encoder_learning_rate = float(
        config["training"].get("speech_encoder_learning_rate", learning_rate)
    )
    text_encoder_learning_rate = float(
        config["training"].get("text_encoder_learning_rate", learning_rate)
    )
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    optimizer_type = config["training"].get("optimizer", "adamw").lower()
    
    # Get beta parameters for Adam/AdamW
    beta1 = float(config["training"].get("beta1", 0.9))
    beta2 = float(config["training"].get("beta2", 0.999))
    
    param_groups = [
        {
            "params": [p for p in adapter.parameters() if p.requires_grad],
            "lr": learning_rate,
            "name": "adapter",
        }
    ]

    if finetune_speech_encoder:
        speech_params = [p for p in speech_encoder.parameters() if p.requires_grad]
        if speech_params:
            param_groups.append(
                {
                    "params": speech_params,
                    "lr": speech_encoder_learning_rate,
                    "name": "speech_encoder",
                }
            )

    if finetune_text_encoder:
        text_params = [p for p in text_encoder.parameters() if p.requires_grad]
        if text_params:
            param_groups.append(
                {
                    "params": text_params,
                    "lr": text_encoder_learning_rate,
                    "name": "text_encoder",
                }
            )

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Use 'adam' or 'adamw'")

    print(f"Optimizer: {optimizer_type.upper()}")
    for group in optimizer.param_groups:
        print(f"  - param_group lr={group['lr']:.2e}, size={len(group['params'])}")

    # LR scheduler (linear warmup + cosine decay by optimizer step)
    batch_size = int(config["training"]["batch_size"])
    gradient_accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    num_epochs = int(config["training"]["num_epochs"])
    num_batches_per_epoch = max(1, math.ceil(len(train_dataset) / batch_size))
    steps_per_epoch = max(1, math.ceil(num_batches_per_epoch / gradient_accumulation_steps))
    total_update_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = int(config["training"].get("warmup_steps", 0))
    scheduler_type = str(config["training"].get("scheduler", "linear_warmup_cosine")).lower()

    scheduler = None
    if scheduler_type != "none":
        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(
            f"Scheduler: {scheduler_type} (total_steps={total_update_steps}, warmup_steps={warmup_steps})"
        )
    
    # Create trainer
    # NOTE: We pass 'collate_fn' here. Ensure your Trainer class accepts it!
    trainer = Trainer(
        text_encoder=text_encoder,
        speech_encoder=speech_encoder,
        adapter=adapter,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=config["paths"]["output_dir"],
        use_wandb=not args.no_wandb,
        project_name="speech-rag",
        collate_fn=speech_collate_fn,  # <--- CRITICAL for padding
        config=config,  # Pass config for wandb logging
        use_amp=bool(config["training"].get("use_amp", True))
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_steps=config["training"].get("save_steps", 1000),
        eval_steps=config["training"].get("eval_steps", 500),
        resume_from=args.resume,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=config["training"].get("early_stopping_patience"),
        early_stopping_min_delta=config["training"].get("early_stopping_min_delta", 0.0),
        log_batch_frequency=config["training"].get("log_batch_frequency", 1)
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()