"""LLM ile Türkçe paraphrase üretimi + etiket kaymasını yakalayan doğrulayıcı.

Etiket (altın JSON) sabit kalır; yalnızca komut cümlesi yeniden yazılır. Bu yüzden her paraphrase
orijinal cümleyle *anlamca eşdeğer* olmalı. LLM'in sık yaptığı hatalar (sayı değiştirme,
yarıçap->çap, uydurma yer/çağrı adı, yön ekleme) kural tabanlı olarak elenir.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from skymind.data.generator import PLACES, numbers_in
from skymind.llm import OpenRouterClient

DEFAULT_MODEL = "google/gemma-3-27b-it"
STYLES = [
    "kısa telsiz emri",
    "günlük konuşma",
    "resmi ve nazik",
    "aceleci, kısa ve eksiltili",
    "yazım hatası/noktalama eksikliği olan hızlı yazım",
    "ayrıntılı ve açıklayıcı",
]

SYSTEM = (
    "Sen bir Türkçe veri artırma aracısın. İHA operatör komutlarını yeniden yazarsın. "
    "KURALLAR: (1) Anlamı birebir koru. (2) Tüm sayıları ve birimleri AYNEN koru "
    "(3 km ise 3 km, 1500 m ise 1500 m). (3) 'yarıçap' kelimesini 'çap'a çevirme. (4) Yön ve yer "
    "adlarını (kuzey, Alfa noktası, baraj...) aynen koru. (5) Yeni bilgi, sayı, yer adı, çağrı adı "
    "ya da ek talimat EKLEME; var olan bilgiyi ÇIKARMA. "
    "(6) Eksik bırakılmış bir bilgiyi tamamlama. "
    "Yalnızca JSON döndür."
)


def build_prompt(commands: Sequence[str], k: int, rng_styles: Sequence[str]) -> str:
    items = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(commands))
    return (
        f"Aşağıdaki {len(commands)} komutun her biri için {k} farklı yeniden yazım üret. "
        f"Kullanabileceğin üsluplar: {', '.join(rng_styles)}.\n\n{items}\n\n"
        f'Çıktı biçimi: {{"results": [[...{k} cümle...], ...]}} '
        f"(komut sırasıyla, toplam {len(commands)} liste)."
    )


# --- doğrulama --------------------------------------------------------------------------------

_DIR_STEMS = ["kuzeydoğu", "kuzeybatı", "güneydoğu", "güneybatı", "kuzey", "güney", "doğu", "batı"]


def _dir_set(text: str) -> set[str]:
    """Metindeki yönler (kuzeydoğu gibi bileşikler tek sayılır)."""
    t = text.lower()
    found: set[str] = set()
    for stem in _DIR_STEMS:  # uzun kökler önce; bulunanı metinden sil
        if stem in t:
            found.add(stem)
            t = t.replace(stem, " ")
    return found


_PLACE_STEMS = {
    "Alfa noktası": "alfa",
    "Bravo noktası": "bravo",
    "Charlie noktası": "charlie",
    "baraj": "baraj",
    "köprü": "köprü",
    "liman": "liman",
    "radar tepesi": "radar",
    "depo": "depo",
    "pist": "pist",
    "sınır karakolu": "karakol",
    "orman sahası": "orman",
}
assert set(_PLACE_STEMS) == set(PLACES)


def _places(text: str) -> set[str]:
    t = text.lower()
    return {name for name, stem in _PLACE_STEMS.items() if stem in t}


def check_paraphrase(original: str, candidate: str) -> str | None:
    """Geçerliyse None, değilse reddetme nedeni."""
    c = candidate.strip()
    if not (3 <= len(c) <= 300) or "\n" in c:
        return "uzunluk/satır"
    if c.lower().rstrip(".!? ") == original.lower().rstrip(".!? "):
        return "aynı"
    if Counter(numbers_in(original)) != Counter(numbers_in(c)):
        return "sayı"
    lo, lc = original.lower(), c.lower()
    if "yarıçap" in lo and not re.search(r"yarı\s?çap", lc):
        return "yarıçap kayboldu"
    if re.search(r"(?<!yarı)(?<!yarı )çap", lc):
        return "çap"
    if _dir_set(original) != _dir_set(c):
        return "yön"
    if _places(original) != _places(c):
        return "yer"
    if ("irtifa" in lo or "yüksek" in lo) != ("irtifa" in lc or "yüksek" in lc):
        return "irtifa ifadesi"
    return None


# --- üretim -----------------------------------------------------------------------------------


_TEXT_KEYS = ("command", "text", "komut", "cümle", "sentence")


def item_text(x: Any) -> str:
    """Bir paraphrase öğesini düz metne çevirir.

    Model bazen cümle yerine ``{"style": ..., "command": ...}`` nesnesi döndürür; ``str(x)`` bunu
    sözlük metnine çevirip komut diye kaydederdi (veride 485 satırı etkiledi). Burada nesneden
    metin alanı çıkarılır; daha önce metne çevrilmiş sözlükler (``"{'style': ...}"``) da onarılır.
    """
    if isinstance(x, dict):
        for k in _TEXT_KEYS:
            if isinstance(x.get(k), str):
                return str(x[k])
        return str(x)
    s = str(x)
    if s.lstrip().startswith("{") and s.rstrip().endswith("}"):
        try:
            d = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s
        if isinstance(d, dict):
            return item_text(d)
    return s


def _parse(text: str) -> list[list[str]] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        res = json.loads(m.group(0))["results"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if isinstance(res, list) and all(isinstance(x, list) for x in res):
        return [[item_text(s) for s in x] for x in res]
    return None


def paraphrase_batch(
    client: OpenRouterClient,
    commands: Sequence[str],
    k: int = 3,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.9,
) -> list[list[str]]:
    """Her komut için doğrulamadan geçmiş paraphrase listesi (boş olabilir)."""
    prompt = build_prompt(commands, k, STYLES)
    resp = client.chat(
        model,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=250 * len(commands),
        temperature=temperature,
    )
    parsed = _parse(resp.text)
    if parsed is None or len(parsed) != len(commands):
        return [[] for _ in commands]
    return [
        list(dict.fromkeys(c for c in cands if check_paraphrase(orig, c) is None))
        for orig, cands in zip(commands, parsed, strict=True)
    ]


def paraphrase_all(
    client: OpenRouterClient,
    rows: Sequence[dict[str, Any]],
    k: int = 3,
    batch_size: int = 8,
    workers: int = 8,
    model: str = DEFAULT_MODEL,
) -> dict[str, list[str]]:
    """id -> geçerli paraphrase listesi."""
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    def run(b: Sequence[dict[str, Any]]) -> list[tuple[str, list[str]]]:
        res = paraphrase_batch(client, [r["command"] for r in b], k, model)
        return [(r["id"], p) for r, p in zip(b, res, strict=True)]

    out: dict[str, list[str]] = {}
    with ThreadPoolExecutor(workers) as ex:
        for pairs in ex.map(run, batches):
            out.update(pairs)
    return out
