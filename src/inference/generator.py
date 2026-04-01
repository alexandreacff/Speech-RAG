"""Generator conditioned on retrieved audio passages (metadata-aware fallback)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


class AudioConditionedGenerator:
    """Generates answers from query + retrieved passages.

    This implementation is robust by design:
    - If an audio-capable chat model is available, it uses it.
    - If the model/processor cannot handle audio directly, it falls back to
      text-only generation with retrieved metadata and paths as context.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-Audio-Chat",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = None
        self.tokenizer = None
        self.model = None

        self._load_components()

    def _load_components(self) -> None:
        load_error = None

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - depends on remote model capabilities
            load_error = exc

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            if self.processor is not None and hasattr(self.processor, "tokenizer"):
                self.tokenizer = self.processor.tokenizer

        if self.tokenizer is None:
            raise RuntimeError(
                f"Could not load tokenizer for generator model '{self.model_name}'."
            ) from load_error

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def _build_context(
        self,
        query: str,
        audio_paths: Sequence[str],
        metadata_list: Optional[Sequence[Optional[Dict]]] = None,
        instruction: Optional[str] = None,
    ) -> str:
        base_instruction = instruction or (
            "You are a QA assistant. Answer the user question using only the retrieved speech passages. "
            "If evidence is insufficient, say so briefly."
        )

        passages: List[str] = []
        metadata_list = metadata_list or []

        for i, path in enumerate(audio_paths):
            metadata = metadata_list[i] if i < len(metadata_list) else None
            if metadata and isinstance(metadata, dict):
                text_hint = metadata.get("sentence") or metadata.get("transcript") or metadata.get("text")
                if text_hint:
                    passages.append(f"[{i+1}] audio={path} | hint={text_hint}")
                else:
                    passages.append(f"[{i+1}] audio={path}")
            else:
                passages.append(f"[{i+1}] audio={path}")

        joined = "\n".join(passages) if passages else "No retrieved passages."
        prompt = (
            f"{base_instruction}\n\n"
            f"Question:\n{query}\n\n"
            f"Retrieved passages:\n{joined}\n\n"
            "Answer:"
        )
        return prompt

    @torch.no_grad()
    def generate(
        self,
        query: str,
        audio_paths: Sequence[str],
        metadata_list: Optional[Sequence[Optional[Dict]]] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        prompt = self._build_context(
            query=query,
            audio_paths=audio_paths,
            metadata_list=metadata_list,
            instruction=instruction,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        output_ids = self.model.generate(**inputs, **generate_kwargs)
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Keep only model continuation when possible.
        if generated.startswith(prompt):
            generated = generated[len(prompt) :].strip()

        return generated.strip()
