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


class LlamaCppPyEngine:
    """llama-cpp-python (aynı llama.cpp çekirdeği, Python bağlaması): C++ servisle aynı GGUF.

    Dili ayrıştırmak için: aynı model dosyası, aynı ChatML istemi, açgözlü örnekleme, bağlam 2048,
    en fazla 400 token ve her istekte sıfırlanan KV önbelleği; yalnızca çevre kod (HTTP, JSON,
    istek döngüsü) Python. Hazırlama/çözme süreleri ayrı ölçülmez (``prefill_ms``/``decode_ms`` 0).
    """

    def __init__(
        self, gguf_path: str, max_new_tokens: int = 400, n_ctx: int = 2048, n_gpu_layers: int = -1
    ) -> None:
        import ctypes
        import os

        from llama_cpp import Llama

        # LLAMA_CPP_LIB_PATH (llama-cpp-python'un kendi değişkeni) başka bir derlemeyi gösteriyorsa
        # (ör. C++ servisin kullandığı dinamik arka uçlu DLL'ler) arka uçları elle yüklemek gerekir.
        lib_dir = os.environ.get("LLAMA_CPP_LIB_PATH")
        if lib_dir and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib_dir)
            ctypes.CDLL(str(Path(lib_dir) / "ggml.dll")).ggml_backend_load_all_from_path(
                lib_dir.encode()
            )
        self.max_new_tokens = max_new_tokens
        self.name = f"llama-cpp-python:{Path(gguf_path).name}"
        self.llm: Any = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            verbose=False,
        )

    def generate(self, system: str, command: str) -> GenerationResult:
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{command}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.llm.reset()  # istekler bağımsız (C++ ile aynı)
        out = self.llm.create_completion(
            prompt, max_tokens=self.max_new_tokens, temperature=0.0, top_k=1, echo=False
        )
        usage = out["usage"]
        return GenerationResult(
            text=out["choices"][0]["text"].strip(),
            prompt_tokens=int(usage["prompt_tokens"]),
            new_tokens=int(usage["completion_tokens"]),
            truncated=out["choices"][0]["finish_reason"] == "length",
        )
