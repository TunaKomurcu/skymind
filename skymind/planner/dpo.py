"""SFT modelinin üstüne DPO (LoRA). Referans model = birleştirilmiş SFT modeli.

SFT adaptörü tabana *birleştirilir*, yeni LoRA bunun üstüne eğitilir; TRL referans log-olasılığı
için yeni adaptörü kapatır -> referans tam olarak SFT modelidir (ham taban değil).

    python -m skymind.planner.dpo --name dpo-v1 --sft-adapter outputs/sft-v1/adapter --subset 3000
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from skymind.planner.preferences import build_pairs
from skymind.planner.prompt import load_split

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    ap.add_argument("--subset", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument(
        "--sft-weight",
        type=float,
        default=0.0,
        help="DPO kaybına eklenen SFT (chosen NLL) ağırlığı; 0 = saf DPO",
    )
    ap.add_argument("--resume", action="store_true", help="son checkpoint'ten devam et")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///outputs/mlflow.db")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "planner-dpo")

    rows = load_split(a.train)
    random.Random(1).shuffle(rows)
    pairs = build_pairs(rows[: a.subset], seed=0)
    ds = Dataset.from_list([{k: v for k, v in p.items() if k != "category"} for p in pairs])

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, a.sft_adapter).merge_and_unload()

    lora = LoraConfig(
        r=a.rank,
        lora_alpha=2 * a.rank,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    out_dir = Path("outputs") / a.name
    cfg = DPOConfig(
        output_dir=str(out_dir),
        run_name=a.name,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        beta=a.beta,
        **(
            {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, a.sft_weight]}
            if a.sft_weight > 0
            else {}
        ),
        bf16=True,
        gradient_checkpointing=True,
        max_length=a.max_length,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to=["mlflow"],
        seed=0,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=True if a.resume else None)
    trainer.save_model(str(out_dir / "adapter"))
    print(f"adaptör kaydedildi: {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
