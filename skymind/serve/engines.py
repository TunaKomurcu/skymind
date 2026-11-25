"""Çıkarım motoru soyutlaması (C++ tarafındaki ``InferenceEngine`` ile aynı sözleşme)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    new_tokens: int = 0
    prefill_ms: float = 0.0  # ilk token'a kadar geçen süre (istem işleme + 1 token)
    decode_ms: float = 0.0  # kalan token'lar
    truncated: bool = False  # max_new_tokens'a takıldı


class Engine(Protocol):
    name: str

    def generate(self, system: str, command: str) -> GenerationResult: ...


class HFEngine:
    """HF transformers, greedy, batch=1. Çağrılar servis katmanında sıraya sokulur."""

    def __init__(self, model_dir: str, max_new_tokens: int = 400, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.name = f"hf-transformers:{Path(model_dir).name}"
        self.tok: Any = AutoTokenizer.from_pretrained(model_dir)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model: Any = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=device)
        model.eval()
        self.model = model

    def generate(self, system: str, command: str) -> GenerationResult:
        from transformers import StoppingCriteria, StoppingCriteriaList

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": command}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        n_prompt = int(enc["input_ids"].shape[1])
        marks: list[float] = []

        class _FirstToken(StoppingCriteria):
            # her token sonrası çağrılır; ilk çağrı = istem işleme + ilk token
            def __call__(self, input_ids: Any, scores: Any, **kw: Any) -> Any:
                if not marks:
                    marks.append(time.perf_counter())
                return False

        t0 = time.perf_counter()
        with self._torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([_FirstToken()]),
            )
        if self.model.device.type == "cuda":
            self._torch.cuda.synchronize()
        t1 = time.perf_counter()
        ids = out[0, n_prompt:]
        n_new = int(ids.shape[0])
        first = (marks[0] if marks else t1) - t0
        return GenerationResult(
            text=self.tok.decode(ids, skip_special_tokens=True).strip(),
            prompt_tokens=n_prompt,
            new_tokens=n_new,
            prefill_ms=first * 1000,
            decode_ms=(t1 - t0 - first) * 1000,
            truncated=n_new >= self.max_new_tokens,
        )
