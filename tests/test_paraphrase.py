import json
from pathlib import Path

import httpx
import pytest

from skymind.data.paraphrase import check_paraphrase, paraphrase_batch
from skymind.llm import OpenRouterClient

ORIG = "Kuzeydeki bölgeyi 3 km yarıçapta 1500 m irtifada tara."


@pytest.mark.parametrize(
    "cand",
    [
        "Kuzey tarafını 3 kilometre yarıçapla, 1500 m irtifada tarar mısın?",  # kilometre != km
    ],
)
def test_unit_rewording_is_rejected_conservatively(cand: str) -> None:
    # sayılar (3, 1500) korunuyor -> sayı kontrolü geçer; 'kilometre' yazımı serbesttir
    assert check_paraphrase(ORIG, cand) is None


@pytest.mark.parametrize(
    ("cand", "reason"),
    [
        ("Kuzeydeki bölgeyi 3 km çapta 1500 m irtifada tara.", "yarıçap kayboldu"),
        ("Kuzeydeki bölgeyi 5 km yarıçapta 1500 m irtifada tara.", "sayı"),
        ("Kuzeydeki bölgeyi 3 km yarıçapta 1500 m irtifada tara, 2 tur at.", "sayı"),
        ("Güneydeki bölgeyi 3 km yarıçapta 1500 m irtifada tara.", "yön"),
        ("Kuzeydeki bölgeyi ve doğuyu 3 km yarıçapta 1500 m irtifada tara.", "yön"),
        ("Kuzeydeki baraj bölgesini 3 km yarıçapta 1500 m irtifada tara.", "yer"),
        ("Kuzeydeki bölgeyi 3 km yarıçapta tara.", "sayı"),
        ("Kuzeydeki bölgeyi 3 km yarıçapta 1500 m'de tara.", None),  # 'irtifa' düşmüş -> kabul
        (ORIG, "aynı"),
        ("x\ny", "uzunluk/satır"),
    ],
)
def test_validator_catches_label_drift(cand: str, reason: str | None) -> None:
    got = check_paraphrase(ORIG, cand)
    if reason is None:
        # 'irtifa' ifadesi kaybı da yakalanmalı
        assert got == "irtifa ifadesi"
    else:
        assert got == reason


def test_compound_directions_not_confused() -> None:
    o = "Kuzeydoğudaki bölgeyi tara"
    assert check_paraphrase(o, "Kuzeydoğu tarafını tarar mısın") is None
    assert check_paraphrase(o, "Kuzeydeki bölgeyi tara") == "yön"


def test_places_must_match() -> None:
    o = "Alfa noktası bölgesini 500 m yarıçapta tara"
    assert check_paraphrase(o, "Alfa noktasının çevresini 500 m yarıçapta tarar mısın") is None
    assert check_paraphrase(o, "Bravo noktası bölgesini 500 m yarıçapta tara") == "yer"


def test_paraphrase_batch_filters_and_handles_bad_json(tmp_path: Path) -> None:
    good = {"results": [["Kuzeydeki bölgeyi 3 km yarıçapta 1500 m irtifada incele", ORIG.upper()]]}
    replies = [json.dumps(good), "bu json değil"]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            data = [{"id": "m", "pricing": {"prompt": "0", "completion": "0"}}]
            return httpx.Response(200, json={"data": data})
        content = replies.pop(0)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}], "usage": {"cost": 0}}
        )

    c = OpenRouterClient(
        api_key="k", budget_usd=1, state_dir=tmp_path, transport=httpx.MockTransport(handler)
    )
    out = paraphrase_batch(c, [ORIG], k=2, model="m")
    assert out == [["Kuzeydeki bölgeyi 3 km yarıçapta 1500 m irtifada incele"]]
    # farklı komut -> farklı önbellek anahtarı; bozuk JSON boş liste döndürür
    assert paraphrase_batch(c, ["Eve dön"], k=2, model="m") == [[]]


def test_item_text_unwraps_dicts_and_repaired_dict_strings() -> None:
    from skymind.data.paraphrase import _parse, item_text

    assert item_text("düz cümle") == "düz cümle"
    assert item_text({"style": "günlük", "command": "Kalk."}) == "Kalk."
    assert (
        item_text("{'style': 'kısa telsiz emri', 'command': 'Doğuya 1km geç.'}")
        == "Doğuya 1km geç."
    )
    assert item_text("{süslü ama sözlük değil}") == "{süslü ama sözlük değil}"
    assert item_text({"başka": 1}) == "{'başka': 1}"
    raw = '{"results": [[{"style": "a", "command": "Birinci."}, "İkinci."]]}'
    assert _parse(raw) == [["Birinci.", "İkinci."]]
