"""Fix skipped sentence indices in audio filenames.

Expected filename format:
	TopicIndex_ParagraphIndex_SentenceIndex.wav

Example:
	0_0_0.wav, 0_0_2.wav, 0_0_3.wav
becomes:
	0_0_0.wav, 0_0_1.wav, 0_0_2.wav
"""

from __future__ import annotations

import argparse
import re
import uuid
from collections import defaultdict
from pathlib import Path


FILENAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.wav$")


def find_audio_files(audio_dir: Path, recursive: bool) -> list[Path]:
	"""Return all .wav files in the directory."""
	pattern = "**/*.wav" if recursive else "*.wav"
	return sorted(audio_dir.glob(pattern))


def parse_filename(path: Path) -> tuple[int, int, int] | None:
	"""Parse topic/paragraph/sentence indices from filename."""
	match = FILENAME_RE.match(path.name)
	if not match:
		return None
	topic_idx, para_idx, sent_idx = match.groups()
	return int(topic_idx), int(para_idx), int(sent_idx)


def build_rename_plan(files: list[Path]) -> tuple[list[tuple[Path, Path]], list[Path]]:
	"""Build rename operations to make sentence indices contiguous per group."""
	grouped: dict[tuple[int, int], list[tuple[int, Path]]] = defaultdict(list)
	ignored: list[Path] = []

	for file_path in files:
		parsed = parse_filename(file_path)
		if parsed is None:
			ignored.append(file_path)
			continue
		topic_idx, para_idx, sent_idx = parsed
		grouped[(topic_idx, para_idx)].append((sent_idx, file_path))

	rename_ops: list[tuple[Path, Path]] = []
	for (topic_idx, para_idx), items in grouped.items():
		# Stable ordering: first by current sentence index, then by name.
		items_sorted = sorted(items, key=lambda x: (x[0], x[1].name))
		for new_sent_idx, (_old_sent_idx, src_path) in enumerate(items_sorted):
			dst_name = f"{topic_idx}_{para_idx}_{new_sent_idx}.wav"
			dst_path = src_path.with_name(dst_name)
			if src_path != dst_path:
				rename_ops.append((src_path, dst_path))

	return rename_ops, ignored


def apply_renames(rename_ops: list[tuple[Path, Path]], dry_run: bool) -> None:
	"""Apply renames using a temporary phase to avoid path collisions."""
	if not rename_ops:
		print("No files need renaming.")
		return

	groups = {(dst.name.split("_")[0], dst.name.split("_")[1]) for src, dst in rename_ops}
	print(f"Planned renames: {len(rename_ops)}")
	print(f"Affected topic/paragraph groups: {len(groups)}")
	print(f"Groups: {groups}")

	for src, dst in rename_ops:
		print(f"  {src.name} -> {dst.name}")

	if dry_run:
		print("Dry run enabled. No files were renamed.")
		return

	temp_ops: list[tuple[Path, Path]] = []
	token = uuid.uuid4().hex[:8]

	# Phase 1: move all sources to temporary names.
	for src, _dst in rename_ops:
		temp_name = f"{src.stem}__tmpfix_{token}.wav"
		temp_path = src.with_name(temp_name)
		src.rename(temp_path)
		temp_ops.append((temp_path, src))

	# Build map from original src to desired dst.
	dst_map = {src: dst for src, dst in rename_ops}

	# Phase 2: move temp names to final destinations.
	for temp_path, original_src in temp_ops:
		final_dst = dst_map[original_src]
		temp_path.rename(final_dst)

	print("Renaming completed.")


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Fix skipped SentenceIndex values in WAV filenames with format "
			"TopicIndex_ParagraphIndex_SentenceIndex.wav"
		)
	)
	parser.add_argument("audio_dir", type=Path, help="Directory containing WAV files")
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Search WAV files recursively in subdirectories",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Only show planned renames without changing files",
	)

	args = parser.parse_args()

	if not args.audio_dir.exists() or not args.audio_dir.is_dir():
		raise ValueError(f"Invalid audio directory: {args.audio_dir}")

	files = find_audio_files(args.audio_dir, recursive=args.recursive)
	if not files:
		print("No .wav files found.")
		return

	rename_ops, ignored = build_rename_plan(files)
	if ignored:
		print(f"Ignored files with unexpected naming: {len(ignored)}")

	apply_renames(rename_ops, dry_run=args.dry_run)


if __name__ == "__main__":
	main()
