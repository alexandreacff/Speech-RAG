"""Trainer structural checks.

Run:
	python test/trainer_check.py
"""

from __future__ import annotations

import ast
import traceback
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT_DIR / "training" / "trainer.py"


def _read_trainer_source() -> str:
	if not TRAINER_PATH.exists():
		raise FileNotFoundError(f"Trainer file not found: {TRAINER_PATH}")
	return TRAINER_PATH.read_text(encoding="utf-8")


def _parse_trainer_ast(source: str) -> ast.Module:
	return ast.parse(source)


def _find_class(tree: ast.Module, class_name: str) -> ast.ClassDef:
	for node in tree.body:
		if isinstance(node, ast.ClassDef) and node.name == class_name:
			return node
	raise AssertionError(f"Class '{class_name}' not found")


def _find_method(cls: ast.ClassDef, method_name: str) -> ast.FunctionDef:
	for node in cls.body:
		if isinstance(node, ast.FunctionDef) and node.name == method_name:
			return node
	raise AssertionError(f"Method '{method_name}' not found in class '{cls.name}'")


def _find_calls(func: ast.FunctionDef, attr_name: str) -> list[ast.Call]:
	calls: list[ast.Call] = []
	for node in ast.walk(func):
		if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
			if node.func.attr == attr_name:
				calls.append(node)
	return calls


def test_trainer_methods_present() -> None:
	source = _read_trainer_source()
	tree = _parse_trainer_ast(source)
	trainer_cls = _find_class(tree, "Trainer")

	required_methods = [
		"_log_trainability_summary",
		"_compute_optimizer_grad_norm",
		"_set_train_modes_for_training",
		"train_epoch",
		"validate",
	]

	for method_name in required_methods:
		_find_method(trainer_cls, method_name)


def test_train_epoch_uses_train_mode_setter() -> None:
	source = _read_trainer_source()
	tree = _parse_trainer_ast(source)
	trainer_cls = _find_class(tree, "Trainer")
	train_epoch = _find_method(trainer_cls, "train_epoch")

	calls = _find_calls(train_epoch, "_set_train_modes_for_training")
	if not calls:
		raise AssertionError("train_epoch should call self._set_train_modes_for_training()")


def test_grad_norm_uses_optimizer_scope() -> None:
	source = _read_trainer_source()
	tree = _parse_trainer_ast(source)
	trainer_cls = _find_class(tree, "Trainer")
	train_epoch = _find_method(trainer_cls, "train_epoch")

	grad_calls = _find_calls(train_epoch, "_compute_optimizer_grad_norm")
	if len(grad_calls) < 2:
		raise AssertionError(
			"Expected _compute_optimizer_grad_norm() in main update and remainder update blocks"
		)

	src_lower = source.lower()
	if "for param in self.adapter.parameters()" in src_lower:
		raise AssertionError(
			"Found adapter-only grad norm loop; grad norm should use optimizer-scoped parameters"
		)


def test_remainder_step_exists() -> None:
	source = _read_trainer_source()
	tree = _parse_trainer_ast(source)
	trainer_cls = _find_class(tree, "Trainer")
	train_epoch = _find_method(trainer_cls, "train_epoch")

	remainder_if_found = False
	for node in ast.walk(train_epoch):
		if isinstance(node, ast.If):
			condition = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
			if "num_batches % gradient_accumulation_steps != 0" in condition:
				remainder_if_found = True
				step_calls = [
					n
					for n in ast.walk(node)
					if isinstance(n, ast.Call)
					and isinstance(n.func, ast.Attribute)
					and n.func.attr == "step"
				]
				if not step_calls:
					raise AssertionError("Remainder block found, but optimizer.step() is missing")

	if not remainder_if_found:
		raise AssertionError("Missing remainder update block for gradient accumulation")


def test_validate_initializes_total_loss() -> None:
	source = _read_trainer_source()
	tree = _parse_trainer_ast(source)
	trainer_cls = _find_class(tree, "Trainer")
	validate = _find_method(trainer_cls, "validate")

	has_total_loss_init = False
	for node in validate.body:
		if isinstance(node, ast.Assign):
			if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
				if node.targets[0].id == "total_loss":
					has_total_loss_init = True
					break

	if not has_total_loss_init:
		raise AssertionError("validate() should initialize total_loss before accumulation")


def warn_on_remainder_scaling_behavior() -> None:
	"""Warn about known behavior: remainder micro-steps are scaled by full accumulation factor."""
	source = _read_trainer_source()

	has_fixed_scaling = "loss = loss / gradient_accumulation_steps" in source
	has_remainder_step = "if num_batches % gradient_accumulation_steps != 0:" in source
	has_remainder_compensation = (
		"remainder_scale = gradient_accumulation_steps / remainder_steps" in source
		and "param.grad.mul_(remainder_scale)" in source
	)

	if has_fixed_scaling and has_remainder_step and not has_remainder_compensation:
		print(
			"[WARN] Remainder accumulation is updated, but final partial window uses "
			"loss scaled by full gradient_accumulation_steps. "
			"This makes the last optimizer step smaller when there is a remainder."
		)
	elif has_fixed_scaling and has_remainder_step and has_remainder_compensation:
		print("[INFO] Remainder scaling compensation detected.")


def main() -> None:
	tests = [
		test_trainer_methods_present,
		test_train_epoch_uses_train_mode_setter,
		test_grad_norm_uses_optimizer_scope,
		test_remainder_step_exists,
		test_validate_initializes_total_loss,
	]

	passed = 0
	failed = 0

	for test_fn in tests:
		name = test_fn.__name__
		try:
			test_fn()
			print(f"[PASS] {name}")
			passed += 1
		except Exception as exc:
			print(f"[FAIL] {name}: {exc}")
			print(traceback.format_exc(limit=1).strip())
			failed += 1

	warn_on_remainder_scaling_behavior()

	print("\nSummary")
	print(f"  Trainer file: {TRAINER_PATH}")
	print(f"  Passed: {passed}")
	print(f"  Failed: {failed}")

	if failed > 0:
		raise SystemExit(1)


if __name__ == "__main__":
	main()
