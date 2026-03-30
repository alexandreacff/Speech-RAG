"""TextEncoder smoke checks focused on output shapes and numerical sanity.

Run examples:
	python test/text_test.py
	python test/text_test.py --model-name Qwen/Qwen3-Embedding-0.6B --device cpu
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

from src.models.text_encoder import TextEncoder


def _assert_shape(name: str, tensor: torch.Tensor, expected_batch: int, expected_dim: int) -> None:
	expected = (expected_batch, expected_dim)
	got = tuple(tensor.shape)
	if got != expected:
		raise AssertionError(f"{name}: expected shape {expected}, got {got}")


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
	if not torch.isfinite(tensor).all():
		raise AssertionError(f"{name}: tensor contains NaN/Inf")


def _assert_normalized(name: str, tensor: torch.Tensor, atol: float = 5e-2) -> None:
	norms = tensor.norm(dim=-1)
	if not torch.allclose(norms, torch.ones_like(norms), atol=atol):
		raise AssertionError(
			f"{name}: expected L2 norms close to 1.0, got min={norms.min().item():.4f}, max={norms.max().item():.4f}"
		)


def _build_batch_texts(batch_size: int) -> list[str]:
	return [
		f"Sample text {idx}: testing text encoder shape consistency and pooling behavior."
		for idx in range(batch_size)
	]


def test_single_text_shape(encoder: TextEncoder, device: str, max_length: int) -> None:
	emb = encoder.encode("Single text input for shape check.", device=device, max_length=max_length, normalize=True)
	_assert_shape("single_text", emb, expected_batch=1, expected_dim=encoder.embedding_dim)
	_assert_finite("single_text", emb)
	_assert_normalized("single_text", emb)
	print(f"[PASS] single_text shape={tuple(emb.shape)}")


def test_batch_text_shape(encoder: TextEncoder, device: str, max_length: int, batch_size: int) -> None:
	texts = _build_batch_texts(batch_size)
	emb = encoder.encode(texts, device=device, max_length=max_length, normalize=True)
	_assert_shape("batch_text", emb, expected_batch=batch_size, expected_dim=encoder.embedding_dim)
	_assert_finite("batch_text", emb)
	_assert_normalized("batch_text", emb)
	print(f"[PASS] batch_text shape={tuple(emb.shape)}")


def test_no_normalize_path(encoder: TextEncoder, device: str, max_length: int) -> None:
	texts = ["No normalize path A", "No normalize path B"]
	emb = encoder.encode(texts, device=device, max_length=max_length, normalize=False)
	_assert_shape("no_normalize", emb, expected_batch=2, expected_dim=encoder.embedding_dim)
	_assert_finite("no_normalize", emb)
	print(f"[PASS] no_normalize shape={tuple(emb.shape)}")


def main() -> None:
	parser = argparse.ArgumentParser(description="TextEncoder shape and numerical sanity checks")
	parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--max-length", type=int, default=128)
	parser.add_argument("--batch-size", type=int, default=4)
	args = parser.parse_args()

	print("[RUN] text encoder checks")
	print(f"  model={args.model_name}")
	print(f"  device={args.device}")

	tests = []
	passed = 0
	failed = 0

	try:
		encoder = TextEncoder(model_name=args.model_name, freeze=True)
		encoder = encoder.to(args.device)
		encoder.eval()
		print(f"  embedding_dim={encoder.embedding_dim}")
	except Exception as exc:
		print(f"[FAIL] encoder_init: {exc}")
		print(traceback.format_exc())
		raise SystemExit(1)

	tests.append(("single_text_shape", lambda: test_single_text_shape(encoder, args.device, args.max_length)))
	tests.append(
		(
			"batch_text_shape",
			lambda: test_batch_text_shape(encoder, args.device, args.max_length, args.batch_size),
		)
	)
	tests.append(("no_normalize_path", lambda: test_no_normalize_path(encoder, args.device, args.max_length)))

	for name, fn in tests:
		try:
			fn()
			passed += 1
		except Exception as exc:
			failed += 1
			print(f"[FAIL] {name}: {exc}")
			print(traceback.format_exc(limit=1).strip())

	print("\nSummary")
	print(f"  Passed: {passed}")
	print(f"  Failed: {failed}")

	if failed > 0:
		raise SystemExit(1)


if __name__ == "__main__":
	main()
