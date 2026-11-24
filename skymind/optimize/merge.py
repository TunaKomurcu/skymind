"""LoRA adaptörlerini tabana birleştirip tek bir HF modeli olarak kaydeder (GGUF/ONNX girdisi).

    python -m skymind.optimize.merge --adapter outputs/sft-v1/adapter --out outputs/merged-sft

CPU'da çalışır (GPU başka işlere kalsın); 1.5B model bf16'da ~3 GB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-adapter", default=None, help="önce birleştirilecek adaptör")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu")
    for path in [p for p in (a.base_adapter, a.adapter) if p]:
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    print(f"kaydedildi: {a.out}")


if __name__ == "__main__":
    main()
