"""Run real train script with real code/models on a tiny generated dataset.

This intentionally ignores other tests and validates the real training entrypoint.

Run:
    python test/all_check.py
    python test/all_check.py --epochs 1 --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
import traceback
import wave
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def _write_wav(path: Path, seconds: float = 0.25, sample_rate: int = 16000) -> None:
    """Write a simple sine-wave wav file without external dependencies."""
    n_samples = max(1, int(seconds * sample_rate))
    freq = 440.0
    amp = 0.2

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for i in range(n_samples):
            value = amp * math.sin(2.0 * math.pi * freq * i / sample_rate)
            sample = int(max(-1.0, min(1.0, value)) * 32767)
            frames.extend(struct.pack("<h", sample))
        wav_file.writeframes(bytes(frames))


def _build_tiny_dataset(data_dir: Path) -> None:
    """Create minimal SpokenSQuAD-like train/val metadata + wav files."""
    train_wav = data_dir / "train_wav"
    dev_wav = data_dir / "dev_wav"
    train_wav.mkdir(parents=True, exist_ok=True)
    dev_wav.mkdir(parents=True, exist_ok=True)

    # The dataset loader expects filenames by index: 0_0_0.wav, 0_0_1.wav
    _write_wav(train_wav / "0_0_0.wav")
    _write_wav(train_wav / "0_0_1.wav")
    _write_wav(dev_wav / "0_0_0.wav")

    train_json = {
        "data": [
            {
                "title": "tiny-train",
                "paragraphs": [
                    {
                        "context": "Tiny context for training check.",
                        "qas": [
                            {"id": "q0", "question": "What is this?", "answers": []},
                            {"id": "q1", "question": "Why run smoke?", "answers": []},
                        ],
                    }
                ],
            }
        ]
    }
    val_json = {
        "data": [
            {
                "title": "tiny-val",
                "paragraphs": [
                    {
                        "context": "Tiny context for validation check.",
                        "qas": [
                            {"id": "q0", "question": "Validation question?", "answers": []}
                        ],
                    }
                ],
            }
        ]
    }

    (data_dir / "spoken_train-v1.1.json").write_text(json.dumps(train_json), encoding="utf-8")
    (data_dir / "spoken_test-v1.1.json").write_text(json.dumps(val_json), encoding="utf-8")


def _load_base_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_runtime_config(base_cfg: Dict[str, Any], data_dir: Path, output_dir: Path, epochs: int) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["training"] = dict(base_cfg.get("training", {}))
    cfg["paths"] = dict(base_cfg.get("paths", {}))
    cfg["data"] = dict(base_cfg.get("data", {}))

    # Keep real model names from base config, but make runtime quick.
    cfg["training"]["num_epochs"] = epochs
    cfg["training"]["batch_size"] = 1
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg["training"]["save_steps"] = 999999
    cfg["training"]["eval_steps"] = 1
    cfg["training"]["log_batch_frequency"] = 1

    cfg["data"]["sample_rate"] = 16000
    cfg["data"]["max_audio_length"] = 1.0

    cfg["paths"]["data_dir"] = str(data_dir)
    cfg["paths"]["output_dir"] = str(output_dir)
    return cfg


def _run_and_stream(command: list[str], cwd: Path, timeout_seconds: int) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
        return process.wait(timeout=timeout_seconds if timeout_seconds > 0 else None)
    except subprocess.TimeoutExpired:
        process.kill()
        _ = process.wait()
        raise TimeoutError(f"Train process exceeded timeout of {timeout_seconds}s")


def run_real_train_check(device: str, epochs: int, timeout_seconds: int) -> None:
    train_script = ROOT_DIR / "scripts" / "train.py"
    base_cfg_path = ROOT_DIR / "config" / "config.yaml"

    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    with tempfile.TemporaryDirectory(prefix="real-train-check-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        data_dir = tmp_dir / "data"
        output_dir = tmp_dir / "outputs"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        _build_tiny_dataset(data_dir)
        base_cfg = _load_base_config(base_cfg_path)
        runtime_cfg = _build_runtime_config(base_cfg, data_dir, output_dir, epochs=epochs)

        runtime_cfg_path = tmp_dir / "runtime_config.yaml"
        runtime_cfg_path.write_text(yaml.safe_dump(runtime_cfg), encoding="utf-8")

        command = [
            sys.executable,
            "-u",
            str(train_script),
            "--config",
            str(runtime_cfg_path),
            "--device",
            device,
            "--no-wandb",
        ]

        print("[RUN]", " ".join(command))
        returncode = _run_and_stream(command=command, cwd=ROOT_DIR, timeout_seconds=timeout_seconds)

        if returncode != 0:
            raise RuntimeError(f"Real train check failed with exit code {returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real train smoke check")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device for train script")
    parser.add_argument("--epochs", type=int, default=1, help="Tiny smoke epochs")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds (0 disables timeout)")
    args = parser.parse_args()

    print("[RUN] scripts/train.py real smoke check")
    try:
        run_real_train_check(device=args.device, epochs=args.epochs, timeout_seconds=args.timeout)
        print("[PASS] scripts/train.py real smoke check")
        print("\nSummary")
        print(f"  Root: {ROOT_DIR}")
        print("  Passed: 1")
        print("  Failed: 0")
    except Exception as exc:
        print(f"[FAIL] scripts/train.py real smoke check: {exc}")
        print(traceback.format_exc())
        print("\nSummary")
        print(f"  Root: {ROOT_DIR}")
        print("  Passed: 0")
        print("  Failed: 1")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
