"""Niyet sınıflandırıcısı için damıtma (distillation) ve budama (pruning) deneyleri.

    python -m skymind.nlp.compress --teacher outputs/nlp/intent-berturk/last

Damıtma: öğrenci, öğretmenin seçili katmanlarından başlatılır. Her boyut için KD'li (yumuşak hedef)
ve KD'siz (yalnızca CE) eğitilir; böylece kazancın damıtmadan mı yoksa yalnızca katman
kırpmadan mı geldiği ayrışır. Budama: yapısal olmayan büyüklük budaması (ağırlıkları sıfırlar) ve
yapısal kafa budaması (gerçekten küçültür). Gecikme yoğun çekirdeklerle ölçülür: seyreklik tek
başına hız kazandırmaz.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F  # noqa: N812
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from skymind.mlops.tracking import log_run
from skymind.nlp.datasets import intent_data
from skymind.nlp.ner_data import INTENTS
from skymind.nlp.train import batches, pad


def encode(
    tok: Any, texts: list[str], labels: list[int], max_len: int = 96
) -> list[dict[str, Any]]:
    enc = tok(texts, truncation=True, max_length=max_len)
    return [{"input_ids": enc["input_ids"][i], "labels": labels[i]} for i in range(len(texts))]


def make_student(teacher: Any, keep: list[int]) -> Any:
    """Öğretmenin ``keep`` katmanlarını (sırayla) alan daha sığ bir model."""
    cfg = copy.deepcopy(teacher.config)
    cfg.num_hidden_layers = len(keep)
    student = type(teacher)(cfg)
    sd = teacher.state_dict()
    new = {k: v for k, v in sd.items() if not re.match(r"bert\.encoder\.layer\.\d+\.", k)}
    for j, i in enumerate(keep):
        pre = f"bert.encoder.layer.{i}."
        for k, v in sd.items():
            if k.startswith(pre):
                new[f"bert.encoder.layer.{j}." + k[len(pre) :]] = v
    student.load_state_dict(new)
    return student


def fit(
    model: Any,
    train: list[dict[str, Any]],
    pad_id: int,
    device: str,
    epochs: int,
    lr: float,
    teacher: Any | None = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    seed: int = 0,
    masks: dict[str, torch.Tensor] | None = None,
) -> float:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + 31) // 32)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps)
    )
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for b in batches(train, 32, True, rng):
            batch = pad(b, pad_id)
            labels = torch.tensor([x["labels"] for x in b]).to(device)
            inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inp).logits
            loss: Any = F.cross_entropy(logits, labels)
            if teacher is not None:
                with torch.inference_mode():
                    t_logits = teacher(**inp).logits
                kd = (
                    F.kl_div(
                        F.log_softmax(logits / temperature, -1),
                        F.softmax(t_logits / temperature, -1),
                        reduction="batchmean",
                    )
                    * temperature**2
                )
                loss = alpha * loss + (1 - alpha) * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if masks:  # seyrek ağırlıklar sıfırda kalsın
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in masks:
                            p.mul_(masks[name])
    return time.perf_counter() - t0


@torch.inference_mode()
def predict(model: Any, tok: Any, texts: list[str], device: str) -> list[int]:
    model.eval()
    out: list[int] = []
    for i in range(0, len(texts), 128):
        enc = tok(
            texts[i : i + 128], truncation=True, max_length=96, padding=True, return_tensors="pt"
        )
        out += model(**{k: v.to(device) for k, v in enc.items()}).logits.argmax(-1).cpu().tolist()
    return out


def score(model: Any, tok: Any, device: str) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    x, y = intent_data("test")
    p = predict(model, tok, x, device)
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
    }


@torch.inference_mode()
def latency_ms(
    model: Any, tok: Any, device: str, texts: list[str], n: int = 150
) -> dict[str, float]:
    """Tek örnek (batch=1) gecikme: ısınma sonrası p50 ve p95, ms."""
    model.eval()
    xs = texts[:n]
    ts: list[float] = []
    for i, t in enumerate(xs + xs[:10]):
        enc = tok(t, truncation=True, max_length=96, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        t0 = time.perf_counter()
        model(**enc)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= 10:
            ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"p50": round(statistics.median(ts), 2), "p95": round(ts[int(0.95 * (len(ts) - 1))], 2)}


def nonzero_params(model: Any) -> float:
    return sum(int((p != 0).sum()) for p in model.parameters()) / 1e6


def report(
    model: Any, tok: Any, device: str, texts: list[str], extra: dict[str, Any]
) -> dict[str, Any]:
    cpu_model = copy.deepcopy(model).to("cpu")
    res = {
        **extra,
        **score(model, tok, device),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "nonzero_params_m": round(nonzero_params(model), 1),
        "latency_gpu_ms": latency_ms(model, tok, device, texts) if device == "cuda" else None,
        "latency_cpu_ms": latency_ms(cpu_model, tok, "cpu", texts),
    }
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


# --- budama ------------------------------------------------------------------------------------


def magnitude_masks(model: Any, sparsity: float) -> dict[str, torch.Tensor]:
    """Kodlayıcıdaki doğrusal katman ağırlıkları üzerinde küresel büyüklük budaması maskeleri."""
    weights = {
        n: p
        for n, p in model.named_parameters()
        if n.startswith("bert.encoder.") and n.endswith(".weight") and p.dim() == 2
    }
    allv = torch.cat([p.detach().abs().flatten() for p in weights.values()])
    k = int(sparsity * allv.numel())
    thr = torch.kthvalue(allv.cpu(), k).values.item()
    return {n: (p.detach().abs() > thr).to(p.dtype) for n, p in weights.items()}


def apply_masks(model: Any, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in masks:
                p.mul_(masks[n])


def head_importance(
    model: Any, items: list[dict[str, Any]], pad_id: int, device: str
) -> torch.Tensor:
    """Michel ve ark.: kafa ölçeği üzerinden |dL/dmask| (birinci derece önem), [katman, kafa].

    transformers 5'te ``head_mask`` parametresi kalktı; bu yüzden her dikkat modülünün çıktısına
    kafa başına ölçek uygulayan bir ileri kancası (forward hook) kullanılır.
    """
    cfg = model.config
    n_heads = cfg.num_attention_heads
    mask = torch.ones(cfg.num_hidden_layers, n_heads, device=device, requires_grad=True)

    def make_hook(li: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> Any:
            ctx = out[0]
            b, s, h = ctx.shape
            scaled = (ctx.view(b, s, n_heads, h // n_heads) * mask[li].view(1, 1, -1, 1)).view(
                b, s, h
            )
            return (scaled, *out[1:])

        return hook

    handles = [
        layer.attention.self.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.bert.encoder.layer)
    ]
    imp = torch.zeros_like(mask)
    model.eval()
    try:
        for b in batches(items, 32, False, random.Random(0)):
            batch = pad(b, pad_id)
            labels = torch.tensor([x["labels"] for x in b]).to(device)
            inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            loss = F.cross_entropy(model(**inp).logits, labels)
            imp += torch.autograd.grad(loss, mask)[0].abs().detach()
    finally:
        for h in handles:
            h.remove()
    return imp


def prune_heads(model: Any, by_layer: dict[int, list[int]]) -> None:
    """Yapısal kafa budaması: Q/K/V çıkış satırlarını ve çıkış projeksiyonu sütunlarını keser.

    ``transformers`` 5'te ``prune_heads`` kalktı. Dikkat modülü kafa sayısını ``view(..., -1, d)``
    ile çıkardığı için doğrusal katmanları küçültmek yeterlidir. ``by_layer`` hâlâ özgün kafa
    numaralarıyla verilir (her katman bir kez budanır).
    """
    cfg = model.config
    d = cfg.hidden_size // cfg.num_attention_heads
    for li, drop in by_layer.items():
        att = model.bert.encoder.layer[li].attention
        keep = [h for h in range(cfg.num_attention_heads) if h not in set(drop)]
        if not keep:
            raise ValueError(f"katman {li}: en az bir kafa kalmalı")
        idx = torch.tensor([h * d + j for h in keep for j in range(d)])
        for name in ("query", "key", "value"):
            old = getattr(att.self, name)
            new = torch.nn.Linear(old.in_features, len(idx)).to(old.weight.device, old.weight.dtype)
            new.weight.data = old.weight.data[idx.to(old.weight.device)].clone()
            new.bias.data = old.bias.data[idx.to(old.bias.device)].clone()
            setattr(att.self, name, new)
        out = att.output.dense
        new_out = torch.nn.Linear(len(idx), out.out_features).to(
            out.weight.device, out.weight.dtype
        )
        new_out.weight.data = out.weight.data[:, idx.to(out.weight.device)].clone()
        new_out.bias.data = out.bias.data.clone()
        att.output.dense = new_out
        att.self.num_attention_heads = len(keep)
        att.self.all_head_size = len(idx)


def choose_heads(imp: torch.Tensor, frac: float) -> dict[int, list[int]]:
    """En önemsiz ``frac`` oranında kafayı küresel seçer; her katmanda en az 1 kafa kalır."""
    n_layers, n_heads = imp.shape
    alive = dict.fromkeys(range(n_layers), n_heads)
    target = int(frac * imp.numel())
    by_layer: dict[int, list[int]] = {}
    dropped = 0
    for idx in torch.argsort(imp.flatten()).tolist():  # en önemsiz önce
        if dropped >= target:
            break
        layer, head = divmod(idx, n_heads)
        if alive[layer] > 1:
            by_layer.setdefault(layer, []).append(head)
            alive[layer] -= 1
            dropped += 1
    return by_layer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="outputs/nlp/intent-berturk/last")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4, help="CPU gecikme ölçümü iş parçacığı")
    ap.add_argument("--out", type=Path, default=Path("outputs/nlp/compress.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.teacher)
    pad_id = tok.pad_token_id
    x_tr, y_tr = intent_data("train")
    train = encode(tok, x_tr, y_tr)
    test_x, _ = intent_data("test")
    teacher: Any = AutoModelForSequenceClassification.from_pretrained(a.teacher).to(device)
    teacher.eval()
    results: list[dict[str, Any]] = []

    def add(r: dict[str, Any]) -> None:
        results.append(r)  # her varyanttan sonra diske yaz: kesilirse kayıp olmasın
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    add(report(teacher, tok, device, test_x, {"variant": "teacher-12L"}))

    def fresh_teacher() -> Any:
        m: Any = AutoModelForSequenceClassification.from_pretrained(a.teacher).to(device)
        return m

    # damıtma
    for keep, name in (([0, 2, 4, 6, 8, 10], "6L"), ([0, 4, 8, 11], "4L"), ([0, 11], "2L")):
        for use_kd in (True, False):
            student = make_student(teacher.cpu(), keep).to(device)
            teacher.to(device)
            secs = fit(student, train, pad_id, device, a.epochs, 5e-5, teacher if use_kd else None)
            add(
                report(
                    student,
                    tok,
                    device,
                    test_x,
                    {
                        "variant": f"student-{name}-{'kd' if use_kd else 'ce'}",
                        "train_s": round(secs, 1),
                    },
                )
            )
            del student

    # yapısal olmayan büyüklük budaması (+ maskeli 1 epoch ince ayar)
    for sp in (0.5, 0.8, 0.9):
        m = fresh_teacher()
        masks = magnitude_masks(m, sp)
        apply_masks(m, masks)
        add(report(m, tok, device, test_x, {"variant": f"magnitude-{int(sp * 100)}-noft"}))
        secs = fit(m, train, pad_id, device, 1, 2e-5, masks=masks)
        add(
            report(
                m,
                tok,
                device,
                test_x,
                {"variant": f"magnitude-{int(sp * 100)}-ft1", "train_s": round(secs, 1)},
            )
        )
        del m

    # yapısal kafa budaması (+ 1 epoch ince ayar)
    m = fresh_teacher()
    imp = head_importance(m, train[:2000], pad_id, device)
    del m
    for frac in (0.3, 0.5, 0.7):
        m = fresh_teacher()
        prune_heads(m, choose_heads(imp, frac))
        results_tag = f"heads-{int(frac * 100)}"
        add(report(m, tok, device, test_x, {"variant": f"{results_tag}-noft"}))
        secs = fit(m, train, pad_id, device, 1, 3e-5)
        add(
            report(
                m, tok, device, test_x, {"variant": f"{results_tag}-ft1", "train_s": round(secs, 1)}
            )
        )
        del m

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        log_run(
            "nlp-compress",
            r["variant"],
            {"teacher": a.teacher, "epochs": a.epochs, "classes": len(INTENTS)},
            {k: v for k, v in r.items() if isinstance(v, float)},
        )
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
