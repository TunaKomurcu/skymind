"""QLoRA ile planlayıcı SFT (TRL SFTTrainer, prompt/completion biçimi).

python -m skymind.planner.sft --name sft-v1 --epochs 2
python -m skymind.planner.sft --name sft-smoke --max-steps 20 --subset 200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from skymind.planner.prompt import load_split, sft_example

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("data/processed/val.jsonl"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--subset", type=int, default=0, help="eğitim örneği sınırı (0 = tümü)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    a = ap.parse_args()

    # ağır importlar burada: CLI --help hızlı kalsın
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///outputs/mlflow.db")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "planner-sft")

    train_rows = load_split(a.train, a.subset or None)
    val_rows = load_split(a.val, 300)
    train_ds = Dataset.from_list([sft_example(r) for r in train_rows])
    val_ds = Dataset.from_list([sft_example(r) for r in val_rows])

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        device_map="cuda",
    )
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
    cfg = SFTConfig(
        output_dir=str(out_dir),
        run_name=a.name,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        max_length=a.max_length,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to=["mlflow"],
        seed=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    print(f"adaptör kaydedildi: {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
