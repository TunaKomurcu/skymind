"""Planlayıcı çıkarımı (HF transformers + opsiyonel LoRA adaptörü). Greedy, toplu üretim."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from skymind.planner.prompt import SYSTEM_PROMPT, messages

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load(
    model_name: str = DEFAULT_MODEL,
    adapter: str | None = None,
    base_adapter: str | None = None,
) -> tuple[Any, Any]:
    """``base_adapter`` önce birleştirilir (DPO için SFT), ``adapter`` onun üstüne."""
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter:
        from peft import PeftModel

        for path in [p for p in (base_adapter, adapter) if p]:
            model = PeftModel.from_pretrained(model, path).merge_and_unload()
    model.eval()
    return tok, model


T = TypeVar("T")


def _batches(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@torch.inference_mode()
def generate(
    tok: Any,
    model: Any,
    commands: Sequence[str],
    batch_size: int = 16,
    max_new_tokens: int = 400,
    system: str = SYSTEM_PROMPT,
) -> list[str]:
    prompts = [
        tok.apply_chat_template(
            messages(c, system=system), tokenize=False, add_generation_prompt=True
        )
        for c in commands
    ]
    # uzunluğa göre sırala: dolgu israfını azaltır; sonra özgün sıraya döndür
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs: dict[int, str] = {}
    for idx in _batches(order, batch_size):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        outs.update(zip(idx, (t.strip() for t in texts), strict=True))
    return [outs[i] for i in range(len(prompts))]


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> str:
    """Hoşgörülü çıkarım: kod bloğu ya da ilk '{' ile son '}' arasını alır."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    i, j = text.find("{"), text.rfind("}")
    return text[i : j + 1] if 0 <= i < j else text
