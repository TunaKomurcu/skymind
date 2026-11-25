"""BERTurk ile niyet sınıflandırma ve NER ince ayarı (basit, yeniden başlatılabilir eğitim döngüsü).

    python -m skymind.nlp.train --task intent --name intent-berturk
    python -m skymind.nlp.train --task ner --name ner-berturk

Her epoch sonunda model ``outputs/nlp/<name>/`` altına kaydedilir; ``--resume`` son epoch'tan
sürdürür. Değerlendirme: niyet doğruluğu/makro-F1; NER varlık F1 (test + elle yazılmış set).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from skymind.mlops.tracking import log_run
from skymind.nlp.datasets import intent_data, manual_ner, ner_data
from skymind.nlp.metrics import entity_f1
from skymind.nlp.ner_data import BIO, INTENTS

DEFAULT_MODEL = "dbmdz/bert-base-turkish-cased"


def encode_ner(
    tok: Any, words: list[list[str]], tags: list[list[str]], max_len: int
) -> list[dict[str, Any]]:
    """Kelime etiketini ilk alt-kelime parçasına yazar; diğerleri -100 (kayıp dışı)."""
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=max_len)
    items = []
    for i, t in enumerate(tags):
        labels, prev = [], None
        for wid in enc.word_ids(i):
            labels.append(-100 if wid is None or wid == prev else BIO.index(t[wid]))
            prev = wid
        items.append({"input_ids": enc["input_ids"][i], "labels": labels})
    return items


def pad(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    lab = torch.full((len(batch), n), -100)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        mask[i, :k] = 1
        if isinstance(b["labels"], list):
            lab[i, :k] = torch.tensor(b["labels"])
    return {"input_ids": ids, "attention_mask": mask, "labels": lab}


def batches(
    items: list[dict[str, Any]], size: int, shuffle: bool, rng: random.Random
) -> list[list[dict[str, Any]]]:
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    return [[items[j] for j in idx[i : i + size]] for i in range(0, len(idx), size)]


@torch.inference_mode()
def predict_ner(model: Any, tok: Any, words: list[list[str]], device: str) -> list[list[str]]:
    model.eval()
    out: list[list[str]] = []
    for i in range(0, len(words), 64):
        chunk = words[i : i + 64]
        enc = tok(chunk, is_split_into_words=True, truncation=True, max_length=96, padding=True)
        ids = torch.tensor(enc["input_ids"]).to(device)
        mask = torch.tensor(enc["attention_mask"]).to(device)
        pred = model(input_ids=ids, attention_mask=mask).logits.argmax(-1).cpu().tolist()
        for j, w in enumerate(chunk):
            tags, prev = ["O"] * len(w), None
            for pos, wid in enumerate(enc.word_ids(j)):
                if wid is not None and wid != prev:
                    tags[wid] = BIO[pred[j][pos]]
                prev = wid
            out.append(tags)
    return out


@torch.inference_mode()
def predict_intent(model: Any, tok: Any, texts: list[str], device: str) -> list[int]:
    model.eval()
    out: list[int] = []
    for i in range(0, len(texts), 128):
        enc = tok(
            texts[i : i + 128], truncation=True, max_length=96, padding=True, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out += model(**enc).logits.argmax(-1).cpu().tolist()
    return out


def evaluate_task(task: str, model: Any, tok: Any, device: str) -> dict[str, Any]:
    if task == "intent":
        from sklearn.metrics import accuracy_score, f1_score

        x, y = intent_data("test")
        p = predict_intent(model, tok, x, device)
        return {
            "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
        }
    res: dict[str, Any] = {}
    w, t = ner_data("test")
    res["test_weak"] = entity_f1(t, predict_ner(model, tok, w, device))
    for sec, (mw, mt) in manual_ner().items():
        res[f"manual_{sec}"] = entity_f1(mt, predict_ner(model, tok, mw, device))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["intent", "ner"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path("outputs/nlp") / a.name
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.task == "intent":
        x, y = intent_data("train")
        enc = tok(x, truncation=True, max_length=a.max_len)
        train = [{"input_ids": enc["input_ids"][i], "labels": y[i]} for i in range(len(x))]
        cls: Any = AutoModelForSequenceClassification
        n_labels = len(INTENTS)
    else:
        w, t = ner_data("train")
        train = encode_ner(tok, w, t, a.max_len)
        cls = AutoModelForTokenClassification
        n_labels = len(BIO)

    start_epoch = 0
    src = a.model
    if a.resume and (out / "last").exists():
        src = str(out / "last")
        start_epoch = json.loads((out / "state.json").read_text())["epoch"]
    model: Any = cls.from_pretrained(src, num_labels=n_labels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps)
    )
    for _ in range(start_epoch * ((len(train) + a.batch - 1) // a.batch)):
        sched.step()

    t0 = time.perf_counter()
    for epoch in range(start_epoch, a.epochs):
        model.train()
        total = 0.0
        bl = batches(train, a.batch, True, rng)
        for b in bl:
            batch = pad(b, tok.pad_token_id)
            if a.task == "intent":
                batch["labels"] = torch.tensor([x["labels"] for x in b])
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss)
        print(f"epoch {epoch + 1}/{a.epochs} loss={total / len(bl):.4f}", flush=True)
        model.save_pretrained(out / "last")
        tok.save_pretrained(out / "last")
        (out / "state.json").write_text(json.dumps({"epoch": epoch + 1}))
    train_s = time.perf_counter() - t0

    res = evaluate_task(a.task, model, tok, device)
    res["train_seconds"] = round(train_s, 1)
    res["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    (out / "metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flat = {
        k: v
        for k, v in (
            {"accuracy": res.get("accuracy"), "macro_f1": res.get("macro_f1")}
            | {
                f"{s}_micro_f1": res[s]["micro"]["f1"]
                for s in res
                if s.startswith(("test_", "manual_"))
            }
        ).items()
        if isinstance(v, float)
    }
    log_run("nlp", a.name, vars(a), flat, tags={"task": a.task})
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    for s in res:
        if s.startswith(("test_", "manual_")):
            print(s, "micro F1", res[s]["micro"]["f1"], "macro F1", res[s]["macro"]["f1"])


if __name__ == "__main__":
    main()
