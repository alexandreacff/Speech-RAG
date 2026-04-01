"""Speech retriever: text query -> nearest audio passages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import faiss
import numpy as np
import torch

from src.models import SpeechAdapter, SpeechEncoder, TextEncoder


class SpeechRetriever:
    """Builds and queries a dense index with text/audio shared embeddings."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        speech_encoder: SpeechEncoder,
        adapter: SpeechAdapter,
        device: Optional[str] = None,
    ) -> None:
        self.text_encoder = text_encoder
        self.speech_encoder = speech_encoder
        self.adapter = adapter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.text_encoder.to(self.device).eval()
        self.speech_encoder.to(self.device).eval()
        self.adapter.to(self.device).eval()

        self.embedding_dim = int(self.adapter.get_embedding_dim())
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.audio_paths: List[str] = []
        self.metadata: List[Optional[Dict]] = []

    def _to_paths(self, audio_files: Sequence[Union[str, Path]]) -> List[str]:
        return [str(Path(p).resolve()) for p in audio_files]

    @torch.no_grad()
    def _encode_text(self, query: str) -> np.ndarray:
        emb = self.text_encoder.encode(query, device=self.device)
        return emb.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _encode_audio_batch(self, audio_batch: Sequence[Union[str, Path]]) -> np.ndarray:
        speech_repr = self.speech_encoder.encode(list(audio_batch), device=self.device)
        speech_emb = self.adapter(speech_repr)
        return speech_emb.detach().cpu().numpy().astype(np.float32)

    def build_index(
        self,
        audio_files: Sequence[Union[str, Path]],
        metadata: Optional[Union[Sequence[Optional[Dict]], Dict[str, Dict]]] = None,
        batch_size: int = 8,
    ) -> None:
        """Build ANN index from audio files.

        Args:
            audio_files: Iterable of audio file paths.
            metadata: Optional metadata per path. If dict, key must be resolved path string.
            batch_size: Batch size for speech embedding extraction.
        """
        paths = self._to_paths(audio_files)
        if not paths:
            raise ValueError("No audio files provided to build index.")

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.audio_paths = []
        self.metadata = []

        if batch_size < 1:
            batch_size = 1

        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch_emb = self._encode_audio_batch(batch_paths)
            all_embeddings.append(batch_emb)

            self.audio_paths.extend(batch_paths)
            if metadata is None:
                self.metadata.extend([None] * len(batch_paths))
            elif isinstance(metadata, dict):
                self.metadata.extend([metadata.get(p) for p in batch_paths])
            else:
                self.metadata.extend(list(metadata[start : start + len(batch_paths)]))

        matrix = np.vstack(all_embeddings).astype(np.float32)
        faiss.normalize_L2(matrix)
        self.index.add(matrix)

    def save_index(self, path: str) -> None:
        """Save FAISS index and side metadata file."""
        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        sidecar = index_path.with_suffix(index_path.suffix + ".meta.json")
        payload = {
            "audio_paths": self.audio_paths,
            "metadata": self.metadata,
            "embedding_dim": self.embedding_dim,
            "index_ntotal": int(self.index.ntotal),
        }
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_index(self, path: str) -> None:
        """Load FAISS index and optional side metadata file."""
        index_path = Path(path)
        self.index = faiss.read_index(str(index_path))

        sidecar = index_path.with_suffix(index_path.suffix + ".meta.json")
        if sidecar.exists():
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.audio_paths = [str(Path(p).resolve()) for p in payload.get("audio_paths", [])]
            self.metadata = payload.get("metadata", [None] * len(self.audio_paths))
        else:
            raise FileNotFoundError(
                f"Missing metadata sidecar for index: {sidecar}. "
                "Rebuild the index with SpeechRetriever.build_index(...) and save_index(...)."
            )

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search the index using a text query."""
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Build or load an index first.")

        k = max(1, min(k, int(self.index.ntotal)))

        qvec = self._encode_text(query)
        faiss.normalize_L2(qvec)

        scores, indices = self.index.search(qvec, k)
        results: List[Dict] = []

        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            idx_int = int(idx)
            audio_path = self.audio_paths[idx_int] if idx_int < len(self.audio_paths) else ""
            metadata = self.metadata[idx_int] if idx_int < len(self.metadata) else None
            results.append(
                {
                    "rank": rank,
                    "index": idx_int,
                    "audio_path": audio_path,
                    "score": float(score),
                    "metadata": metadata,
                }
            )

        return results
