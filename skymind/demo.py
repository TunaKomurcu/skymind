"""Uçtan uca demo: planlayıcı servisi + METAR/NOTAM özeti + atıflı kılavuz yanıtı.

    python -m skymind.demo --url http://127.0.0.1:8091         # planlayıcı servis çalışırken
    python -m skymind.demo --skip-planner                      # model gerektirmeyen kısımlar

Üç bölüm bağımsızdır: (1) Türkçe komut -> doğrulanmış JSON plan / netleştirme / ret (servis ister),
(2) METAR ve NOTAM'ın kural tabanlı Türkçe özeti (model yok), (3) kılavuzdan cümle atıflı yanıt
(BM25, model yok). Komutlar elle yazıldı; eğitim verisinden bilerek farklı, çıktılar gösterim
amaçlıdır (doğruluk iddiası değildir, ölçüm için docs/RESULTS.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx

COMMANDS = [
    "Kalkıştan sonra 120 metre irtifaya çık, ardından barajı 2 km yarıçapta tara.",
    "İHA'yı kuzeye doğru 800 metre götür ve oradan geri dön.",
    "Hızı yirmi beş m/s yap.",  # sayı sözcüğü: zor örnek
    "Oraya git.",  # eksik bilgi -> netleştirme beklenir
    "Havalimanı pistine alçak geçiş yap, irtifa 5 metre.",  # güvensiz -> ret beklenir
]
METAR = "METAR LTBA 261050Z 22012G22KT 4000 -RA BKN015 08/06 Q1008 NOSIG"
NOTAM = (
    "A0123/25 NOTAMN\nQ) LTAA/QWULW/IV/NBO/A/000/999/4059N02845E005\n"
    "A) LTBA B) 2503151000 C) 2503161000\nD) DAILY 0800-1600\n"
    "E) UAS OPR WI 3NM RADIUS 4059N 02845E. MAX 400FT AGL\nF) SFC G) 400FT AGL"
)
QUESTIONS = [
    "X1 modelinde pervane kaç uçuş saatinde bir değiştirilir?",
    "X2 modelinde batarya hücre gerilimi şarj tamamlandığında kaç volt olmalıdır?",
]


def _head(title: str) -> None:
    print(f"\n=== {title} ===")


def _plan_line(step: dict[str, Any]) -> str:
    extra = {k: v for k, v in step.items() if k != "action"}
    return f"  - {step['action']}: {json.dumps(extra, ensure_ascii=False)}"


def show_planner(url: str, timeout: float, transport: httpx.BaseTransport | None = None) -> int:
    _head("1) Planlayıcı (Türkçe komut -> doğrulanmış JSON)")
    bad = 0
    with httpx.Client(base_url=url, timeout=timeout, transport=transport) as c:
        for cmd in COMMANDS:
            print(f"\nKomut: {cmd}")
            r = c.post("/v1/plan", json={"command": cmd})
            r.raise_for_status()
            d = r.json()
            tag = f"[{d['type']}]" if d["valid"] else f"[GEÇERSİZ: {d['validation_error']}]"
            print(f"  {tag}  {d['latency_ms']:.0f} ms, {d['usage']['new_tokens']} token")
            bad += not d["valid"]
            plan = d.get("plan") or {}
            if plan.get("type") == "plan":
                for s in plan["steps"]:
                    print(_plan_line(s))
            elif d["valid"]:
                print("  " + json.dumps(plan, ensure_ascii=False))
            else:
                print("  ham çıktı: " + d["raw"][:200])
    print(f"\nGeçersiz çıktı: {bad}/{len(COMMANDS)} (şema doğrulaması sunucu tarafında yapılır)")
    return bad


def show_aviation() -> None:
    from skymind.aviation.metar import advisory_tr, parse_metar, summarize_tr
    from skymind.aviation.notam import parse_notam
    from skymind.aviation.notam import summarize_tr as notam_tr

    _head("2) METAR / NOTAM (kural tabanlı, model yok)")
    m = parse_metar(METAR)
    print(f"\n{METAR}\n  {summarize_tr(m)}")
    for w in advisory_tr(m) or ["prototip eşik aşımı yok (uçuş izni anlamına GELMEZ)"]:
        print(f"  uyarı: {w}")
    print(f"\n{NOTAM}\n  {notam_tr(parse_notam(NOTAM))}")


def show_rag() -> None:
    from skymind.rag.answer import extractive_answer, verify
    from skymind.rag.corpus import build_corpus
    from skymind.rag.retrieval import BM25, ranking

    _head("3) Kılavuzdan atıflı yanıt (BM25 + cümle seçimi, model yok; kurgusal kılavuz)")
    corpus = build_corpus(3)
    bm = BM25([c.text for c in corpus.chunks], stem5=True)
    for q in QUESTIONS:
        order = ranking(bm.scores(q))[:3]
        cites = extractive_answer(q, corpus.chunks, order, max_sentences=1)
        ok = verify([(c.chunk_id, c.sentence_idx) for c in cites], corpus.chunks)
        print(f"\nSoru: {q}")
        for c, v in zip(cites, ok, strict=True):
            print(f"  {c.sentence}  {c}  (atıf geçerli: {v})")
    print("\nNot: yanıtlayıcı yalnızca ilk cümleyi seçer; modeller benzer cümleler paylaştığı için")
    print("model karışıklığı mümkündür (ölçüm: docs/RESULTS.md, Faz 5).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--skip-planner", action="store_true")
    a = ap.parse_args()
    bad = 0
    if not a.skip_planner:
        try:
            bad = show_planner(a.url, a.timeout)
        except httpx.HTTPError as e:
            print(f"Planlayıcı servise ulaşılamadı ({a.url}): {e}")
            print("--skip-planner ile model gerektirmeyen kısımlar çalıştırılabilir.")
            return 2
    show_aviation()
    show_rag()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
