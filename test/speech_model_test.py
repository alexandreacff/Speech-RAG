"""SpeechEncoder shape and compatibility checks.

Run examples:
	python test/speech_model_test.py
	python test/speech_model_test.py --device cpu --batch-size 4 --audio-seconds 2.0
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
import sys

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.models import SpeechAdapter, SpeechEncoder


def _assert_3d(name: str, x: torch.Tensor) -> None:
	if x.ndim != 3:
		raise AssertionError(f"{name}: expected 3D tensor [batch, seq_len, hidden], got {tuple(x.shape)}")


def _assert_finite(name: str, x: torch.Tensor) -> None:
	if not torch.isfinite(x).all():
		raise AssertionError(f"{name}: contains NaN/Inf")


def _assert_batch_size(name: str, x: torch.Tensor, expected_batch: int) -> None:
	got = x.shape[0]
	if got != expected_batch:
		raise AssertionError(f"{name}: expected batch={expected_batch}, got batch={got}, shape={tuple(x.shape)}")


def _build_audio(batch_size: int, sample_rate: int, audio_seconds: float, device: str) -> torch.Tensor:
	n_samples = max(1, int(sample_rate * audio_seconds))
	return torch.randn(batch_size, n_samples, device=device, dtype=torch.float32)


def _run_case(name: str, fn) -> tuple[bool, str]:
	try:
		fn()
		print(f"[PASS] {name}")
		return True, ""
	except Exception as exc:
		print(f"[FAIL] {name}: {exc}")
		print(traceback.format_exc(limit=1).strip())
		return False, str(exc)


def main() -> None:
	parser = argparse.ArgumentParser(description="SpeechEncoder shape checks")
	parser.add_argument("--model-name", type=str, default="facebook/hubert-large-ls960-ft")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--sample-rate", type=int, default=16000)
	parser.add_argument("--audio-seconds", type=float, default=1.0)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--freeze", action="store_true", help="Initialize SpeechEncoder frozen")
	args = parser.parse_args()

	print("[RUN] speech encoder checks")
	print(f"  model={args.model_name}")
	print(f"  device={args.device}")
	print(f"  sample_rate={args.sample_rate}")

	try:
		speech_encoder = SpeechEncoder(
			model_name=args.model_name,
			freeze=args.freeze,
			target_sample_rate=args.sample_rate,
		).to(args.device)
		if args.freeze:
			speech_encoder.eval()
		else:
			speech_encoder.train()
	except Exception as exc:
		print(f"[FAIL] encoder_init: {exc}")
		print(traceback.format_exc())
		raise SystemExit(1)

	base_audio = _build_audio(args.batch_size, args.sample_rate, args.audio_seconds, args.device)
	single_audio = base_audio[0]
	mono_audio = single_audio.unsqueeze(0)
	audio_list = [base_audio[i].detach().cpu() for i in range(args.batch_size)]

	test_results = []

	def case_single_1d() -> None:
		out = speech_encoder.encode(single_audio, device=args.device)
		_assert_3d("single_1d", out)
		_assert_batch_size("single_1d", out, 1)
		_assert_finite("single_1d", out)
		print(f"  single_1d shape={tuple(out.shape)}")

	def case_single_mono_2d() -> None:
		out = speech_encoder.encode(mono_audio, device=args.device)
		_assert_3d("single_mono_2d", out)
		_assert_batch_size("single_mono_2d", out, 1)
		_assert_finite("single_mono_2d", out)
		print(f"  single_mono_2d shape={tuple(out.shape)}")

	def case_batch_2d() -> None:
		print(f"Testing batch input with shape {tuple(base_audio.shape)}")
		out = speech_encoder.encode(base_audio, device=args.device)
		_assert_3d("batch_2d", out)
		_assert_batch_size("batch_2d", out, args.batch_size)
		_assert_finite("batch_2d", out)
		print(f"  batch_2d shape={tuple(out.shape)}")

	def case_list_input() -> None:
		out = speech_encoder.encode(audio_list, device=args.device)
		_assert_3d("list_input", out)
		_assert_batch_size("list_input", out, args.batch_size)
		_assert_finite("list_input", out)
		print(f"  list_input shape={tuple(out.shape)}")

	def case_adapter_compatibility() -> None:
		out = speech_encoder.encode(base_audio, device=args.device)
		hidden_dim = out.shape[-1]
		adapter = SpeechAdapter(input_dim=hidden_dim, output_dim=1024, downsample_factor=4).to(args.device)
		adapter_out = adapter(out)
		if adapter_out.ndim != 2:
			raise AssertionError(f"adapter output expected 2D [batch, dim], got {tuple(adapter_out.shape)}")
		_assert_batch_size("adapter_out", adapter_out, args.batch_size)
		_assert_finite("adapter_out", adapter_out)
		print(f"  adapter_out shape={tuple(adapter_out.shape)}")

	cases = [
		("single_1d", case_single_1d),
		("single_mono_2d", case_single_mono_2d),
		("batch_2d", case_batch_2d),
		("list_input", case_list_input),
		("adapter_compatibility", case_adapter_compatibility),
	]

	passed = 0
	for name, fn in cases:
		ok, _ = _run_case(name, fn)
		test_results.append((name, ok))
		if ok:
			passed += 1

	failed = len(cases) - passed
	print("\nSummary")
	print(f"  Passed: {passed}")
	print(f"  Failed: {failed}")

	if failed > 0:
		failed_names = [name for name, ok in test_results if not ok]
		print(f"  Failed cases: {', '.join(failed_names)}")
		raise SystemExit(1)


if __name__ == "__main__":
	main()
