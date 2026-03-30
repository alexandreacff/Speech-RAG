"""VRAM guard test for worst-case training step.

Run examples:
	python test/test_mem.py
	python test/test_mem.py --config config/config.yaml --safety-ratio 0.90
	python test/test_mem.py --batch-size 4 --audio-length-multiplier 1.2

This script performs one conservative synthetic training step in CUDA and
checks peak VRAM usage against a configurable safety ratio.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import torch
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.models import SpeechAdapter, SpeechEncoder, TextEncoder
from training.losses import DistillationLoss


def _load_config(config_path: Path) -> Dict:
	with open(config_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def _format_gb(num_bytes: float) -> float:
	return float(num_bytes) / (1024.0 ** 3)


def _build_optimizer(
	adapter: SpeechAdapter,
	speech_encoder: SpeechEncoder,
	text_encoder: TextEncoder,
	config: Dict,
) -> torch.optim.Optimizer:
	training_cfg = config.get("training", {})
	learning_rate = float(training_cfg.get("learning_rate", 5e-5))
	speech_lr = float(training_cfg.get("speech_encoder_learning_rate", learning_rate))
	text_lr = float(training_cfg.get("text_encoder_learning_rate", learning_rate))
	weight_decay = float(training_cfg.get("weight_decay", 0.01))
	beta1 = float(training_cfg.get("beta1", 0.9))
	beta2 = float(training_cfg.get("beta2", 0.999))
	optimizer_type = str(training_cfg.get("optimizer", "adamw")).lower()

	finetune_speech_encoder = bool(training_cfg.get("finetune_speech_encoder", True))
	finetune_text_encoder = bool(training_cfg.get("finetune_text_encoder", False))

	param_groups: List[Dict] = [
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
					"lr": speech_lr,
					"name": "speech_encoder",
				}
			)

	if finetune_text_encoder:
		text_params = [p for p in text_encoder.parameters() if p.requires_grad]
		if text_params:
			param_groups.append(
				{
					"params": text_params,
					"lr": text_lr,
					"name": "text_encoder",
				}
			)

	if optimizer_type == "adam":
		return torch.optim.Adam(param_groups, betas=(beta1, beta2), weight_decay=weight_decay)
	return torch.optim.AdamW(param_groups, betas=(beta1, beta2), weight_decay=weight_decay)


def run_vram_guard(
	config_path: Path,
	safety_ratio: float,
	batch_size_override: int | None,
	audio_length_multiplier: float,
	force_no_amp: bool,
) -> None:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA não disponível. Sem GPU não é possível validar VRAM de treino.")

	config = _load_config(config_path)
	data_cfg = config.get("data", {})
	training_cfg = config.get("training", {})

	device = "cuda"
	sample_rate = int(data_cfg.get("sample_rate", 16000))
	max_audio_length = float(data_cfg.get("max_audio_length", 60.0))
	batch_size = int(batch_size_override or training_cfg.get("batch_size", 2))
	use_amp_cfg = bool(training_cfg.get("use_amp", True))
	use_amp = bool(use_amp_cfg and (not force_no_amp))

	num_samples = int(math.ceil(sample_rate * max_audio_length * audio_length_multiplier))
	if num_samples <= 0:
		raise ValueError("num_samples inválido. Verifique sample_rate/max_audio_length/multiplier.")

	finetune_text_encoder = bool(training_cfg.get("finetune_text_encoder", False))
	finetune_speech_encoder = bool(training_cfg.get("finetune_speech_encoder", True))

	text_encoder = TextEncoder(
		model_name=config["models"]["text_encoder"],
		freeze=not finetune_text_encoder,
	).to(device)
	speech_encoder = SpeechEncoder(
		model_name=config["models"]["speech_encoder"],
		freeze=not finetune_speech_encoder,
	).to(device)
	adapter = SpeechAdapter(
		input_dim=speech_encoder.hidden_size,
		output_dim=text_encoder.embedding_dim,
		downsample_factor=4,
	).to(device)

	optimizer = _build_optimizer(adapter, speech_encoder, text_encoder, config)
	loss_fn = DistillationLoss(loss_type=str(training_cfg.get("loss_type", "cosine"))).to(device)

	adapter.train(any(p.requires_grad for p in adapter.parameters()))
	speech_encoder.train(any(p.requires_grad for p in speech_encoder.parameters()))
	text_encoder.train(any(p.requires_grad for p in text_encoder.parameters()))

	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
	autocast_ctx = (
		torch.autocast(device_type="cuda", dtype=torch.float16)
		if use_amp
		else torch.cuda.amp.autocast(enabled=False)
	)

	# Synthetic worst-case-ish batch: all audios at max configured length.
	audio = torch.randn(batch_size, num_samples, device=device, dtype=torch.float32)
	texts = [
		"worst case memory guard query"
		for _ in range(batch_size)
	]

	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()
	optimizer.zero_grad(set_to_none=True)

	with autocast_ctx:
		speech_reprs = speech_encoder.encode(audio, device=device)
		text_embeddings = text_encoder.encode(texts, device=device)
		audio_embeddings = adapter(speech_reprs)
		loss = loss_fn(audio_embeddings, text_embeddings)

	if use_amp:
		scaler.scale(loss).backward()
		scaler.unscale_(optimizer)
		scaler.step(optimizer)
		scaler.update()
	else:
		loss.backward()
		optimizer.step()

	optimizer.zero_grad(set_to_none=True)
	torch.cuda.synchronize()

	peak_allocated = torch.cuda.max_memory_allocated()
	peak_reserved = torch.cuda.max_memory_reserved()
	total_vram = torch.cuda.get_device_properties(0).total_memory
	reserved_ratio = float(peak_reserved) / float(total_vram)

	print("[INFO] VRAM guard report")
	print(f"  config: {config_path}")
	print(f"  batch_size: {batch_size}")
	print(f"  audio_seconds: {num_samples / sample_rate:.2f}")
	print(f"  use_amp: {use_amp}")
	print(f"  peak_allocated_gb: {_format_gb(peak_allocated):.2f}")
	print(f"  peak_reserved_gb: {_format_gb(peak_reserved):.2f}")
	print(f"  total_vram_gb: {_format_gb(total_vram):.2f}")
	print(f"  reserved_ratio: {reserved_ratio:.3f}")
	print(f"  safety_ratio: {safety_ratio:.3f}")

	if reserved_ratio > safety_ratio:
		raise RuntimeError(
			"VRAM guard FAILED: pico de memória acima do limite de segurança. "
			f"ratio={reserved_ratio:.3f} > safety_ratio={safety_ratio:.3f}"
		)


def main() -> None:
	parser = argparse.ArgumentParser(description="VRAM guard test for training")
	parser.add_argument(
		"--config",
		type=str,
		default="config/config.yaml",
		help="Path para config de treino",
	)
	parser.add_argument(
		"--safety-ratio",
		type=float,
		default=0.90,
		help="Limite de segurança de VRAM reservada (0-1)",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=None,
		help="Override de batch size para teste",
	)
	parser.add_argument(
		"--audio-length-multiplier",
		type=float,
		default=1.0,
		help="Multiplicador sobre max_audio_length para estressar pior caso",
	)
	parser.add_argument(
		"--force-no-amp",
		action="store_true",
		help="Desliga AMP para um teste mais conservador de VRAM",
	)
	args = parser.parse_args()

	config_path = ROOT_DIR / args.config
	if not config_path.exists():
		raise FileNotFoundError(f"Config não encontrada: {config_path}")

	if not (0.0 < args.safety_ratio < 1.0):
		raise ValueError("--safety-ratio deve estar entre 0 e 1")

	if args.batch_size is not None and args.batch_size <= 0:
		raise ValueError("--batch-size deve ser > 0")

	if args.audio_length_multiplier <= 0:
		raise ValueError("--audio-length-multiplier deve ser > 0")

	try:
		run_vram_guard(
			config_path=config_path,
			safety_ratio=args.safety_ratio,
			batch_size_override=args.batch_size,
			audio_length_multiplier=args.audio_length_multiplier,
			force_no_amp=args.force_no_amp,
		)
		print("[PASS] VRAM guard")
	except Exception as exc:
		print(f"[FAIL] VRAM guard: {exc}")
		print(traceback.format_exc(limit=1).strip())
		raise SystemExit(1)


if __name__ == "__main__":
	main()
