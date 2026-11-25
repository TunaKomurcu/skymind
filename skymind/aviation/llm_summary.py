"""LLM ile METAR/NOTAM özeti ve kural tabanlı özetleyiciyle sadakat karşılaştırması.

    python -m skymind.aviation.llm_summary --models google/gemma-3-27b-it,google/gemma-3-4b-it

Kaynak raporlar sentetik üreticiden gelir (altın yapı bilinir). Her özet üç ölçüyle denetlenir:

* **Eksik bilgi (recall):** altın alanlardan hangileri özette geçiyor (rüzgâr, görüş, sıcaklık, QNH,
  bulut yüksekliği; NOTAM'da konum, numara, geçerlilik tarihleri, metindeki sayılar).
* **Uydurma sayı:** kaynaktan türetilemeyen sayı (``unsupported_numbers``). Alt sınırdır:
  kaynaktaki bir sayıyla çakışan yanlış sayı yakalanmaz.
* **Anlamsal hata (dar):** rüzgâr yönü sözcüğü (kuzey/güneybatı...) altın dereceyle çelişiyor mu;
  NOTAM'da "kapalı / servis dışı / yasak" durumu atlanmış mı.

Bu ölçüler anlamın tamamını denetlemez (yanlış hadise, yanlış sıra, eksik uyarı görülmez).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from skymind.aviation.faithfulness import (
    numbers_in,
    source_numbers_metar,
    source_numbers_notam,
    unsupported_numbers,
)
from skymind.aviation.generate import metar_text, notam_text, random_metar, random_notam
from skymind.aviation.metar import CARDINALS, Metar, parse_metar
from skymind.aviation.metar import summarize_tr as metar_summary
from skymind.aviation.notam import Notam, parse_notam
from skymind.aviation.notam import summarize_tr as notam_summary
from skymind.rag.retrieval import turkish_lower

SYSTEM = (
    "Sen bir İHA operasyon asistanısın. Verilen havacılık raporunu operatör için kısa, düz Türkçe "
    "bir özete çevir. YALNIZCA rapordaki bilgiyi kullan; sayıları aynen koru, birim çevirme, "
    "kendi yorumunu veya tahminini ekleme. Raporda bulunan hiçbir değeri atlama."
)
FEWSHOT_SYSTEM = (
    SYSTEM + "\n\nOkuma kuralları:\n"
    "METAR: rüzgâr 'ddd ss KT' = ilk 3 hane derece yönü, son 2 hane knot hızı (22012G22KT: 220°, 12 kt, "  # noqa: E501
    "hamle 22 kt); görüş 4 hane metre (0800 = 800 m, 9999 = 10 km ve üzeri); bulut 'FEW/SCT/BKN/OVC' + "  # noqa: E501
    "3 hane yüz fit (030 = 3000 ft, 008 = 800 ft); 'M' eksi demektir (M07/M12 = -7°C / -12°C); "
    "Q1008 = QNH 1008 hPa; 'ggssddZ' = ayın günü, saat, dakika (UTC).\n"
    "NOTAM: ilk satır numara; 'A)' havaalanı ICAO kodudur (Q satırındaki ilk kod FIR'dır, havaalanı DEĞİL); "  # noqa: E501
    "'B)' başlangıç, 'C)' bitiş = YYAAGGSSdd UTC (2503151000 = 15.03.2025 10:00); 'D)' saat programı; "  # noqa: E501
    "'E)' metin; 'F)/G)' alt/üst irtifa. CLSD = kapalı, U/S = servis dışı.\n"
    "Tarihleri gg.aa.yyyy ss:dd UTC biçiminde yaz; tüm sayıları ve ICAO kodlarını koru."
)
FEWSHOT_EXAMPLES = [
    (
        "METAR",
        "METAR LTAC 151320Z 31008KT 9999 SCT040 M02/M06 Q1021",
        "LTAC, ayın 15'i 13:20 UTC: rüzgâr 310° yönünden 8 kt; görüş 10 km veya üzeri; 4000 ft'te "
        "az bulutlu (SCT); sıcaklık -2°C, çiğ noktası -6°C; QNH 1021 hPa.",
    ),
    (
        "NOTAM",
        "B0457/25 NOTAMN\nQ) LTAA/QMRLC/IV/NBO/A/000/999/3930N03230E005\nA) LTAC B) 2509020600 "
        "C) 2509041800\nE) RWY 03/21 CLSD DUE TO MAINT\nF) SFC G) UNL",
        "NOTAM B0457/25 (yeni), havaalanı LTAC: 02.09.2025 06:00 – 04.09.2025 18:00 UTC arasında pist "  # noqa: E501
        "03/21 bakım nedeniyle kapalı. İrtifa: yüzey – sınırsız.",
    ),
]
_CARD_RE = re.compile(r"(kuzeydoğu|kuzeybatı|güneydoğu|güneybatı|kuzey|güney|doğu|batı)")
STATUS = {  # NOTAM metin kalıbı -> özette beklenen kök(ler)
    "CLSD": ("kapal",),
    "U/S": ("servis dışı", "hizmet dışı", "kullanım dışı", "kullanılamaz", "çalışmıyor", "arıza"),
    "PROHIBITED": ("yasak",),
}


def _has(summary_nums: list[float], value: float) -> bool:
    return any(abs(x - value) < 1e-6 for x in summary_nums)


def _signed_present(summary: str, value: int) -> bool:
    """Sıcaklık: mutlak değer geçmeli; negatifse önünde eksi işareti/“eksi” olmalı."""
    n = abs(value)
    if value >= 0:
        return _has(numbers_in(summary), n)
    return bool(re.search(rf"(?:[-−–]\s?|eksi\s+){n}(?!\d)", turkish_lower(summary)))


def metar_facts(m: Metar, summary: str) -> dict[str, bool]:
    nums = numbers_in(summary)
    facts: dict[str, bool] = {}
    if m.wind_speed_kt:
        ms = round(m.wind_speed_kt * 0.514444)
        facts["wind_speed"] = _has(nums, m.wind_speed_kt) or _has(nums, ms)
        if m.wind_dir and not m.wind_variable:
            facts["wind_dir"] = _has(nums, m.wind_dir) or bool(_CARD_RE.search(summary.lower()))
    if m.gust_kt:
        facts["gust"] = _has(nums, m.gust_kt) or _has(nums, round(m.gust_kt * 0.514444))
    if not m.cavok and m.visibility_m is not None:
        v = m.visibility_m
        facts["visibility"] = (
            _has(nums, v) or _has(nums, v / 1000) or (v >= 9999 and _has(nums, 10))
        )
    if m.cavok:
        facts["cavok"] = "cavok" in summary.lower() or _has(nums, 10)
    for i, c in enumerate(m.clouds):
        if c.height_ft:
            facts[f"cloud{i}"] = _has(nums, c.height_ft) or _has(nums, c.height_ft / 1000)
    if m.temp_c is not None:
        facts["temp"] = _signed_present(summary, m.temp_c)
    if m.dew_c is not None:
        facts["dew"] = _signed_present(summary, m.dew_c)
    if m.qnh_hpa is not None:
        facts["qnh"] = _has(nums, m.qnh_hpa)
    return facts


def wind_direction_contradiction(m: Metar, summary: str) -> bool:
    """Özet bir yön sözcüğü söylüyor ama hiçbiri altın yönle (45° sektör) örtüşmüyor."""
    if not m.wind_dir or m.wind_variable:
        return False
    # rüzgâr dışı yön sözcükleri de olabilir (ör. "bulut batıya"); yalnızca rüzgâr cümlesine bak
    wind_clause = next((c for c in re.split(r"[;.\n]", summary) if "rüzgâr" in c.lower()), "")
    found = set(_CARD_RE.findall(wind_clause.lower()))
    if not found:
        return False
    gold = CARDINALS[round(m.wind_dir / 45) % 8]
    return gold not in found


MONTHS = ("ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül",
          "ekim", "kasım", "aralık")  # fmt: skip


def _date_present(summary: str, nums: list[float], d: datetime) -> bool:
    """Tarih-saat özette geçiyor mu: ham YYAAGGSSdd kodu, ya da gün + saat + (ay sayısı|ay adı)."""
    if d.strftime("%y%m%d%H%M") in summary:
        return True
    month = _has(nums, float(d.month)) or MONTHS[d.month - 1] in turkish_lower(summary)
    return month and _has(nums, float(d.day)) and _has(nums, float(d.hour))


def notam_facts(n: Notam, summary: str) -> dict[str, bool]:
    nums = numbers_in(summary)
    facts: dict[str, bool] = {"ident": _has(nums, float(n.number))}
    for loc in n.locations:
        facts[f"loc:{loc}"] = loc.lower() in summary.lower()  # turkish_lower "I"->"ı" yapar
    for tag, d in (("from", n.valid_from), ("to", None if n.permanent else n.valid_to)):
        if d:
            facts[f"{tag}_date"] = _date_present(summary, nums, d)
    for i, x in enumerate(numbers_in(n.text)):
        facts[f"text_num{i}"] = _has(nums, x)
    return facts


def notam_status_missed(n: Notam, summary: str) -> bool:
    low = turkish_lower(summary)
    for key, stems in STATUS.items():
        if key in n.text.upper() and not any(turkish_lower(s) in low for s in stems):
            return True
    return False


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / max(1, len(xs)), 3)


def evaluate_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def agg(kind: str) -> dict[str, Any]:
        rs = [r for r in rows if r["kind"] == kind]
        return {
            "n": len(rs),
            "fact_recall_mean": _mean([r["recall"] for r in rs]),
            "all_facts_present": _mean([float(r["recall"] == 1.0) for r in rs]),
            "with_unsupported_number": _mean([float(bool(r["unsupported"])) for r in rs]),
            "semantic_error": _mean([float(r["semantic_error"]) for r in rs]),
            "mean_chars": round(sum(len(r["summary"]) for r in rs) / max(1, len(rs))),
        }

    return {"metar": agg("metar"), "notam": agg("notam")}


def build_cases(n_metar: int, n_notam: int, seed: int = 21) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for _ in range(n_metar):
        text = metar_text(random_metar(rng))
        cases.append({"kind": "metar", "text": text})
    for _ in range(n_notam):
        cases.append({"kind": "notam", "text": notam_text(random_notam(rng)[0])})
    return cases


def score_case(kind: str, text: str, summary: str) -> dict[str, Any]:
    if kind == "metar":
        m = parse_metar(text)
        facts = metar_facts(m, summary)
        bad = wind_direction_contradiction(m, summary)
        unsup = unsupported_numbers(summary, source_numbers_metar(m))
    else:
        n = parse_notam(text)
        facts = notam_facts(n, summary)
        bad = notam_status_missed(n, summary)
        unsup = unsupported_numbers(summary, source_numbers_notam(n))
    return {
        "kind": kind,
        "text": text,
        "summary": summary,
        "recall": _mean([float(v) for v in facts.values()]) if facts else 1.0,
        "missing": sorted(k for k, v in facts.items() if not v),
        "unsupported": unsup,
        "semantic_error": bad,
    }


def run(
    cases: list[dict[str, Any]], summarize: Callable[[str, str], str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [score_case(c["kind"], c["text"], summarize(c["kind"], c["text"])) for c in cases]
    return rows, evaluate_summaries(rows)


def rule_based(kind: str, text: str) -> str:
    return metar_summary(parse_metar(text)) if kind == "metar" else notam_summary(parse_notam(text))


def build_messages(kind: str, text: str, prompt: str = "zeroshot") -> list[dict[str, str]]:
    label = "METAR" if kind == "metar" else "NOTAM"
    user = f"{label} raporu:\n{text}\n\nÖzet:"
    if prompt == "zeroshot":
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if prompt != "fewshot":
        raise ValueError(f"bilinmeyen prompt: {prompt}")
    msgs = [{"role": "system", "content": FEWSHOT_SYSTEM}]
    for lab, src, out in FEWSHOT_EXAMPLES:
        msgs += [
            {"role": "user", "content": f"{lab} raporu:\n{src}\n\nÖzet:"},
            {"role": "assistant", "content": out},
        ]
    return msgs + [{"role": "user", "content": user}]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="zeroshot", help="virgülle: zeroshot,fewshot")
    ap.add_argument("--models", default="google/gemma-3-27b-it,google/gemma-3-4b-it")
    ap.add_argument("--n-metar", type=int, default=80)
    ap.add_argument("--n-notam", type=int, default=48)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval/summ-llm.json"))
    a = ap.parse_args()

    from skymind.llm import OpenRouterClient

    client = OpenRouterClient()
    cases = build_cases(a.n_metar, a.n_notam, a.seed)
    result: dict[str, Any] = {}
    rows, summ = run(cases, rule_based)
    result["rule-based"] = {"summary": summ, "rows": rows}
    for model in a.models.split(","):
        for prompt in a.prompts.split(","):

            def ask(kind: str, text: str, model: str = model, prompt: str = prompt) -> str:
                msgs = build_messages(kind, text, prompt)
                return client.chat(model, msgs, max_tokens=300, temperature=0.0).text.strip()

            rows, summ = run(cases, ask)
            result[f"{model}|{prompt}"] = {"summary": summ, "rows": rows}
    for name, v in result.items():
        print(name, json.dumps(v["summary"], ensure_ascii=False))
    result["spent_usd_total"] = round(client.ledger.spent(), 4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {a.out} (toplam harcama: {result['spent_usd_total']} USD)")


if __name__ == "__main__":
    main()
