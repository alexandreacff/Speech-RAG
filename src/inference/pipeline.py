"""Speech RAG pipeline: retrieval + generation orchestration."""

from __future__ import annotations

from typing import Dict, Optional

from .generator import AudioConditionedGenerator
from .retriever import SpeechRetriever


class SpeechRAGPipeline:
    """End-to-end helper to retrieve audio passages and generate an answer."""

    def __init__(
        self,
        retriever: SpeechRetriever,
        generator: AudioConditionedGenerator,
        top_k_audio: int = 3,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k_audio = int(max(1, top_k_audio))

    def retrieve_and_generate(
        self,
        query: str,
        k: Optional[int] = None,
        return_retrieval_results: bool = False,
        **generation_kwargs,
    ) -> Dict:
        top_k = int(max(1, k if k is not None else self.top_k_audio))
        retrieval_results = self.retriever.search(query, k=top_k)

        audio_paths = [item["audio_path"] for item in retrieval_results]
        metadata_list = [item.get("metadata") for item in retrieval_results]

        response = self.generator.generate(
            query=query,
            audio_paths=audio_paths,
            metadata_list=metadata_list,
            **generation_kwargs,
        )

        output = {
            "query": query,
            "response": response,
            "audio_paths": audio_paths,
            "num_audios": len(audio_paths),
        }

        if return_retrieval_results:
            output["retrieval_results"] = retrieval_results

        return output
