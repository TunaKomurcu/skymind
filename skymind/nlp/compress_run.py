"""Damıtma ve budama deneylerinin çalıştırıcısı (niyet sınıflandırıcısı).

    python -m skymind.nlp.compress_run --phase acc --seeds 0,1
    python -m skymind.nlp.compress_run --phase latency      # sessiz makinede, eğitimsiz

İki aşama bilerek ayrıdır:

* ``acc``: doğruluk. Öğrenciler ÖNCEDEN EĞİTİLMİŞ BERT katmanlarından başlar (öğretmenle aynı
  3 epoch'luk bütçe); öğretmenin ince ayarlı katmanlarından başlatmak "öğretmen + fazladan eğitim"
  olur ve karşılaştırmayı bozar. Budamalı satırların adil kıyası için "teacher-ft1" kontrolü
  (öğretmen + 1 epoch) vardır. Sonuçlar ``(varyant, tohum)`` bazında yazılır ve yeniden
  başlatılabilir (biten satırlar atlanır).
* ``latency``: gecikme yalnızca mimariye bağlıdır, ağırlıklara değil; bu yüzden eğitimsiz
  mimariler ölçülür. Her biri 3 kez ölçülüp en küçük p50 alınır (müdahaleye dayanıklı).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from skymind.mlops.tracking import log_run
from skymind.nlp.compress import (
    apply_masks,
    choose_heads,
    encode,
    fit,
    head_importance,
    latency_ms,
    magnitude_masks,
    make_student,
    nonzero_params,
    prune_heads,
    score,
)
from skymind.nlp.datasets import intent_data
from skymind.nlp.ner_data import INTENTS

BASE = "dbmdz/bert-base-turkish-cased"
STUDENTS = (([0, 2, 4, 6, 8, 10], "6L"), ([0, 4, 8, 11], "4L"), ([0, 11], "2L"))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def accuracy_phase(a: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.teacher)
    pad_id = tok.pad_token_id
    x_tr, y_tr = intent_data("train")
    train = encode(tok, x_tr, y_tr)
    rows: list[dict[str, Any]] = (
        json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else []
    )
    done = {(r["variant"], r["seed"]) for r in rows}

    def fresh_teacher() -> Any:
        m: Any = AutoModelForSequenceClassification.from_pretrained(a.teacher).to(device)
        return m

    def add(name: str, seed: int, model: Any, **extra: Any) -> None:
        r = {
            "variant": name,
            "seed": seed,
            **score(model, tok, device),
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "nonzero_params_m": round(nonzero_params(model), 1),
            **extra,
        }
        rows.append(r)
        _write(a.out, rows)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    teacher = fresh_teacher().eval()
    if ("teacher-12L", -1) not in done:
        add("teacher-12L", -1, teacher)
    imp = head_importance(fresh_teacher(), train[:2000], pad_id, device)
    base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(INTENTS))

    for seed in a.seeds:
        torch.manual_seed(seed)

        def todo(name: str, seed: int = seed) -> bool:
            return (name, seed) not in done

        if todo("teacher-ft1"):
            m = fresh_teacher()
            add(
                "teacher-ft1",
                seed,
                m,
                train_s=round(fit(m, train, pad_id, device, 1, 2e-5, seed=seed), 1),
            )
            del m
        for keep, size in STUDENTS:
            for kd in (True, False):
                name = f"student-{size}-{'kd' if kd else 'ce'}"
                if not todo(name):
                    continue
                st = make_student(base, keep)
                st.classifier.weight.data.normal_(
                    0, 0.02
                )  # tohuma göre farklı sınıflandırıcı başlangıcı
                st.classifier.bias.data.zero_()
                st = st.to(device)
                secs = fit(
                    st, train, pad_id, device, a.epochs, 5e-5, teacher if kd else None, seed=seed
                )
                add(name, seed, st, train_s=round(secs, 1))
                del st
        for sp in (0.5, 0.8):
            name = f"magnitude-{int(sp * 100)}-ft1"
            if todo(name):
                m = fresh_teacher()
                masks = magnitude_masks(m, sp)
                apply_masks(m, masks)
                secs = fit(m, train, pad_id, device, 1, 2e-5, masks=masks, seed=seed)
                add(name, seed, m, train_s=round(secs, 1))
                del m
        for frac in (0.3, 0.5, 0.7):
            name = f"heads-{int(frac * 100)}-ft1"
            if todo(name):
                m = fresh_teacher()
                prune_heads(m, choose_heads(imp, frac))
                secs = fit(m, train, pad_id, device, 1, 3e-5, seed=seed)
                add(name, seed, m, train_s=round(secs, 1))
                del m
    for r in rows:
        log_run(
            "nlp-compress",
            f"{r['variant']}-s{r['seed']}",
            {"teacher": a.teacher, "epochs": a.epochs, "seed": r["seed"]},
            {k: v for k, v in r.items() if isinstance(v, float)},
        )


def latency_phase(a: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.teacher)
    x_tr, y_tr = intent_data("train")
    train = encode(tok, x_tr[:2000], y_tr[:2000])
    texts, _ = intent_data("test")

    def teacher_cpu() -> Any:
        m: Any = AutoModelForSequenceClassification.from_pretrained(a.teacher)
        return m.eval()

    imp = head_importance(teacher_cpu().to(device), train, tok.pad_token_id, device)
    variants: dict[str, Any] = {"teacher-12L": teacher_cpu()}
    for keep, size in STUDENTS:
        variants[f"student-{size}"] = make_student(teacher_cpu(), keep).eval()
    for frac in (0.3, 0.5, 0.7):
        m = teacher_cpu()
        prune_heads(m, choose_heads(imp, frac))
        variants[f"heads-{int(frac * 100)}"] = m.eval()
    m = teacher_cpu()
    apply_masks(m, magnitude_masks(m, 0.8))
    variants["magnitude-80 (yoğun çekirdek)"] = m.eval()

    out: list[dict[str, Any]] = []
    for name, model in variants.items():
        cpu = [latency_ms(model, tok, "cpu", texts)["p50"] for _ in range(3)]
        gpu = None
        if device == "cuda":
            g = model.to("cuda")
            gpu = min(latency_ms(g, tok, "cuda", texts)["p50"] for _ in range(3))
            model.to("cpu")
        row = {
            "variant": name,
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "latency_cpu_p50_ms": min(cpu),
            "latency_cpu_runs_ms": cpu,
            "latency_gpu_p50_ms": gpu,
            "cpu_threads": a.threads,
        }
        out.append(row)
        _write(a.out, out)
        print(json.dumps(row, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["acc", "latency"], required=True)
    ap.add_argument("--teacher", default="outputs/nlp/intent-berturk/last")
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    a.out = a.out or Path(f"outputs/nlp/compress-{a.phase}.json")
    (accuracy_phase if a.phase == "acc" else latency_phase)(a)


if __name__ == "__main__":
    main()
