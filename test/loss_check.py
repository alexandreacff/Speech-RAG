"""Cosine-loss sanity checks for DistillationLoss.

Run:
	python test/loss_check.py
"""

from __future__ import annotations

import traceback

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.losses import DistillationLoss


def _check_close(name: str, value: torch.Tensor, target: float, tol: float = 1e-5) -> None:
	scalar = float(value.detach().cpu().item())
	if abs(scalar - target) > tol:
		raise AssertionError(f"{name}: expected ~{target}, got {scalar}")


def _check_finite(name: str, value: torch.Tensor) -> None:
	if not torch.isfinite(value).all():
		raise AssertionError(f"{name}: contains NaN or Inf")


def test_cosine_identical() -> None:
	loss_fn = DistillationLoss(loss_type="cosine", normalize_for_cosine=True)
	x = torch.randn(4, 8)
	loss = loss_fn(x, x)
	print(f"Cosine loss for identical vectors: {loss.item()}")
	_check_close("cosine identical", loss, 0.0, tol=1e-5)


def test_cosine_opposite() -> None:
	loss_fn = DistillationLoss(loss_type="cosine", normalize_for_cosine=True)
	x = torch.randn(4, 8)
	loss = loss_fn(x, -x)
	print(f"Cosine loss for opposite vectors: {loss.item()}")
	_check_close("cosine opposite", loss, 2.0, tol=1e-4)


def test_shape_mismatch_raises() -> None:
	loss_fn = DistillationLoss(loss_type="cosine")
	a = torch.randn(4, 8)
	b = torch.randn(4, 7)
	try:
		_ = loss_fn(a, b)
	except AssertionError:
		return
	raise AssertionError("shape mismatch should raise AssertionError")


def test_zero_vector_stability() -> None:
	loss_fn = DistillationLoss(loss_type="cosine", normalize_for_cosine=True)
	a = torch.zeros(4, 8)
	b = torch.randn(4, 8)
	loss = loss_fn(a, b)
	_check_finite("zero vector stability", loss)


def test_compute_similarity_api() -> None:
	"""Detect regression in compute_similarity implementation."""
	loss_fn = DistillationLoss(loss_type="cosine", normalize_for_cosine=True)
	a = torch.randn(4, 8)
	b = torch.randn(4, 8)
	sim = loss_fn.compute_similarity(a, b)
	if sim.shape != (4,):
		raise AssertionError(f"compute_similarity shape expected (4,), got {tuple(sim.shape)}")
	_check_finite("compute_similarity", sim)


def main() -> None:
	tests = [
		test_cosine_identical,
		test_cosine_opposite,
		test_shape_mismatch_raises,
		test_zero_vector_stability,
		test_compute_similarity_api,
	]

	passed = 0
	failed = 0

	for test_fn in tests:
		name = test_fn.__name__
		try:
			test_fn()
			print(f"[PASS] {name}")
			passed += 1
		except Exception as exc:  # Keep all failures visible in one run.
			print(f"[FAIL] {name}: {exc}")
			print(traceback.format_exc(limit=1).strip())
			failed += 1

	print("\nSummary")
	print(f"  Passed: {passed}")
	print(f"  Failed: {failed}")

	if failed > 0:
		raise SystemExit(1)


if __name__ == "__main__":
	main()
