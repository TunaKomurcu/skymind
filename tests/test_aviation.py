"""METAR/NOTAM ayrıştırma, özet ve sadakat denetimi."""

import random

import pytest

from skymind.aviation.faithfulness import (
    numbers_in,
    source_numbers_metar,
    source_numbers_notam,
    unsupported_numbers,
)
from skymind.aviation.generate import metar_text, random_metar, random_notam
from skymind.aviation.metar import Limits, advisory_tr, parse_metar, summarize_tr
from skymind.aviation.notam import extract_entities, parse_notam
from skymind.aviation.notam import summarize_tr as notam_summary

# El yazımı örnekler (üreticiden bağımsız beklentiler)
LTBA = "METAR LTBA 251350Z 03008KT 9999 FEW030 SCT100 12/05 Q1019 NOSIG"
LTFM = (
    "LTFM 251420Z 18018G32KT 140V220 3000 -RA BKN008 OVC015CB 09/08 Q0998 TEMPO 1500 +TSRA BKN005CB"
)


def test_parse_basic_metar() -> None:
    m = parse_metar(LTBA)
    assert (m.station, m.day, m.hour, m.minute) == ("LTBA", 25, 13, 50)
    assert (m.wind_dir, m.wind_speed_kt, m.gust_kt) == (30, 8.0, None)
    assert m.visibility_m == 9999 and not m.cavok
    assert [(c.amount, c.height_ft) for c in m.clouds] == [("FEW", 3000), ("SCT", 10000)]
    assert (m.temp_c, m.dew_c, m.qnh_hpa, m.trend) == (12, 5, 1019, "NOSIG")
    assert m.unparsed == []


def test_parse_gust_variable_wind_trend_does_not_overwrite_main_report() -> None:
    m = parse_metar(LTFM)
    assert (m.wind_dir, m.wind_speed_kt, m.gust_kt) == (180, 18.0, 32.0)
    assert m.visibility_m == 3000  # TEMPO'daki 1500 ana görüşü ezmemeli
    assert m.weather == ["-RA"] and [c.amount for c in m.clouds] == ["BKN", "OVC"]
    assert m.trend == "TEMPO" and m.trend_detail == ["1500", "+TSRA", "BKN005CB"]


def test_parse_negative_temps_cavok_vrb_and_qnh_inhg() -> None:
    m = parse_metar("SPECI LTAC 251455Z VRB02KT CAVOK M02/M05 Q1031")
    assert m.wind_variable and m.wind_dir is None and m.cavok and (m.temp_c, m.dew_c) == (-2, -5)
    us = parse_metar("LTAC 010000Z 27010KT 9999 SKC 20/10 A2992")
    assert us.qnh_hpa == 1013 and us.sky_clear


def test_mps_wind_converted_to_knots() -> None:
    m = parse_metar("LTAC 010000Z 27005MPS 9999 20/10 Q1013")
    assert m.wind_speed_kt == pytest.approx(9.7, abs=0.1)


def test_unknown_groups_are_reported_not_dropped_and_rmk_ignored() -> None:
    m = parse_metar("LTAC 010000Z 27010KT 9999 ZZZ99 20/10 Q1013 RMK QWERTY 555")
    assert m.unparsed == ["ZZZ99"]


@pytest.mark.parametrize("bad", ["", "LTBA", "LTBA 999999Z", "12345 251350Z", "LTBA 322500Z"])
def test_parse_metar_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_metar(bad)


def test_summary_and_advisory() -> None:
    s = summarize_tr(parse_metar(LTBA))
    assert "kuzeydoğu (30°)" in s and "8 kt (≈4 m/s)" in s and "QNH 1019 hPa" in s
    adv = advisory_tr(parse_metar(LTFM))
    assert any("16 m/s" in a for a in adv) and any("görüş 3 km" in a for a in adv)
    assert advisory_tr(parse_metar(LTBA)) == []
    strict = advisory_tr(parse_metar(LTBA), Limits(max_wind_ms=3))
    assert strict and "4 m/s" in strict[0]
    assert any(
        "tehlikeli" in a for a in advisory_tr(parse_metar("LTAC 010000Z 00000KT 2000 TSRA Q1/"))
    )


def test_metar_roundtrip_on_random_structures() -> None:
    rng = random.Random(7)
    for _ in range(400):
        gold = random_metar(rng)
        got = parse_metar(metar_text(gold))
        assert got.unparsed == []
        for f in (
            "station",
            "day",
            "hour",
            "minute",
            "report_type",
            "auto",
            "wind_dir",
            "wind_variable",
            "wind_speed_kt",
            "gust_kt",
            "visibility_m",
            "cavok",
            "weather",
            "clouds",
            "temp_c",
            "dew_c",
            "qnh_hpa",
            "trend",
        ):
            assert getattr(got, f) == getattr(gold, f), (f, metar_text(gold))


def test_rule_based_metar_summaries_are_numerically_faithful() -> None:
    rng = random.Random(11)
    for _ in range(300):
        m = parse_metar(metar_text(random_metar(rng)))
        extra = unsupported_numbers(summarize_tr(m), source_numbers_metar(m))
        assert extra == [], (m.raw, extra)


def test_faithfulness_checker_catches_injected_hallucinations() -> None:
    m = parse_metar(LTBA)
    allowed = source_numbers_metar(m)
    good = summarize_tr(m)
    assert unsupported_numbers(good, allowed) == []
    assert unsupported_numbers(good.replace("1019", "1013"), allowed) == [1013.0]
    assert unsupported_numbers(good + " Hamle 33 kt.", allowed) == [33.0]
    # bilinen sınır: kaynaktaki herhangi bir sayıyla (burada gün=25) rastlantıyla çakışan uydurma
    # sayı yakalanmaz; denetim bir alt sınırdır, tam sadakat ölçüsü değil
    assert unsupported_numbers(good + " Hamle 25 kt.", allowed) == []
    assert unsupported_numbers(good.replace("12°C", "15°C"), allowed) == [15.0]


def test_numbers_in_splits_dates_and_times() -> None:
    assert numbers_in("25.11.2025 08:00 – 01.12.2025 16:00") == [
        25,
        11,
        2025,
        8,
        0,
        1,
        12,
        2025,
        16,
        0,
    ]
    assert numbers_in("0800-1600 ve 118.500") == [800, 1600, 118.5]


NOTAM = """A1234/25 NOTAMN
Q) LTAA/QMRLC/IV/NBO/A/000/999/4059N02845E005
A) LTBA B) 2511250800 C) 2512011600
D) DAILY 0800-1600
E) RWY 06/24 CLSD DUE TO MAINT. TWY A2 U/S. CONTACT 118.500 MHZ
F) SFC G) UNL"""


def test_parse_notam_fields() -> None:
    n = parse_notam(NOTAM)
    assert n.ident == "A1234/25" and n.action == "NOTAMN" and n.locations == ["LTBA"]
    assert n.q is not None and (n.q.code, n.q.lower_fl, n.q.upper_fl) == ("QMRLC", 0, 999)
    assert n.q.center == "4059N02845E" and n.q.radius_nm == 5
    assert (n.valid_from.isoformat(), n.valid_to.isoformat()) == (  # type: ignore[union-attr]
        "2025-11-25T08:00:00",
        "2025-12-01T16:00:00",
    )
    assert n.schedule == "DAILY 0800-1600" and (n.lower, n.upper) == ("SFC", "UNL")


def test_notam_entities_and_summary() -> None:
    n = parse_notam(NOTAM)
    ent = extract_entities(n.text)
    assert (
        ent["runway"] == ["06/24"] and ent["taxiway"] == ["A2"] and ent["frequency"] == ["118.500"]
    )
    s = notam_summary(n)
    assert "konu: pist, durum: kapalı (QMRLC)" in s and "nedeniyle bakım" in s
    assert "25.11.2025 08:00 – 01.12.2025 16:00" in s and "yüzey – sınırsız" in s


def test_notam_permanent_estimated_and_unknown_codes_are_honest() -> None:
    perm = parse_notam(
        "B0001/25 NOTAMN\nQ) LTAA/QXXXX/IV/NBO/A/000/999\nA) LTBA B) 2501010000 C) PERM\nE) TEST"
    )
    assert perm.permanent and "süresiz" in notam_summary(perm)
    assert "QXXXX" in notam_summary(perm) and "konu: SX" not in notam_summary(perm)
    est = parse_notam("B0002/25 NOTAMR B0001/25\nA) LTBA B) 2501010000 C) 2501020000EST\nE) X")
    assert est.estimated_end and est.replaces == "B0001/25" and "(tahmini)" in notam_summary(est)


@pytest.mark.parametrize("bad", ["", "foo", "A1234/25 NOTAMN\nB) 2511250800"])
def test_parse_notam_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_notam(bad)


def test_notam_roundtrip_and_numeric_faithfulness_on_random_structures() -> None:
    rng = random.Random(5)
    for _ in range(300):
        gold, text = random_notam(rng)
        got = parse_notam(text)
        assert (got.ident, got.action, got.replaces) == (gold.ident, gold.action, gold.replaces)
        assert got.locations == gold.locations and got.text == gold.text
        assert (got.valid_from, got.valid_to) == (gold.valid_from, gold.valid_to)
        assert got.estimated_end == gold.estimated_end and got.schedule == gold.schedule
        assert got.q is not None and gold.q is not None and got.q.code == gold.q.code
        assert (got.q.center, got.q.radius_nm) == (gold.q.center, gold.q.radius_nm)
        assert unsupported_numbers(notam_summary(got), source_numbers_notam(got)) == [], text


# --- LLM özetleme değerlendirmesi (ağsız) ------------------------------------------------------


def test_summary_metrics_give_rule_based_summaries_a_perfect_score() -> None:
    """Metrik doğrulaması: kural tabanlı özetleyici hiçbir ölçüde ceza almamalı."""
    from skymind.aviation import llm_summary as ls

    rows, summ = ls.run(ls.build_cases(60, 40, seed=4), ls.rule_based)
    for kind in ("metar", "notam"):
        assert summ[kind]["fact_recall_mean"] == 1.0, [r for r in rows if r["missing"]][:2]
        assert summ[kind]["with_unsupported_number"] == 0.0
        assert summ[kind]["semantic_error"] == 0.0


def test_summary_metrics_catch_mutated_summaries() -> None:
    from skymind.aviation import llm_summary as ls

    text = "METAR LTBA 121050Z 22012G22KT 4000 -RA BKN015 M03/M05 Q1008"
    ok = ls.score_case(
        "metar",
        text,
        "LTBA: rüzgâr güneybatıdan 12 kt, hamle 22 kt; görüş 4 km; bulut 1500 ft; "
        "sıcaklık -3°C, çiğ noktası -5°C; QNH 1008.",
    )
    assert ok["recall"] == 1.0 and not ok["unsupported"] and not ok["semantic_error"]
    # yön çelişkisi
    bad_dir = ls.score_case("metar", text, "Rüzgâr kuzeyden 12 kt, hamle 22 kt.")
    assert bad_dir["semantic_error"] is True
    # eksi işareti düşmüş: sıcaklık +3 gibi okunur -> bilgi eksik sayılır
    no_sign = ls.score_case("metar", text, "Sıcaklık 3°C, çiğ noktası 5°C, QNH 1008, rüzgâr 12 kt.")
    assert "temp" in no_sign["missing"] and "dew" in no_sign["missing"]
    # uydurma sayı + eksik QNH
    fab = ls.score_case("metar", text, "Rüzgâr 12 kt, görüş 4000 m, bulut 1700 ft, hamle 22 kt.")
    assert 1700.0 in fab["unsupported"] and "qnh" in fab["missing"]


def test_notam_status_polarity_and_dates_are_checked() -> None:
    from skymind.aviation import llm_summary as ls

    text = (
        "A0123/25 NOTAMN\nQ) LTAA/QMRLC/IV/NBO/A/000/999/4059N02845E005\n"
        "A) LTBA B) 2503151000 C) 2503161000\nE) RWY 09/27 CLSD DUE TO MAINT\nF) SFC G) UNL"
    )
    good = ls.score_case(
        "notam",
        text,
        "LTBA, 123 numaralı NOTAM: 15.03.2025 10:00 - 16.03.2025 10:00 pist 09/27 kapalı.",
    )
    assert good["recall"] == 1.0 and good["semantic_error"] is False
    bad = ls.score_case("notam", text, "LTBA, 123: 15.03 10:00 - 16.03 10:00 pist 09/27 bakımda.")
    assert bad["semantic_error"] is True


def test_faithfulness_accepts_reformatted_times_and_years_but_not_new_numbers() -> None:
    from skymind.aviation import llm_summary as ls

    text = (
        "A0123/25 NOTAMN\nQ) LTAA/QMRLC/IV/NBO/A/000/999/4059N02845E005\n"
        "A) LTBA B) 2503151000 C) 2503161000\nD) DAILY 0800-1600\nE) RWY 09/27 CLSD\nF) SFC G) UNL"
    )
    ok = (
        "LTBA 123: 15 Mart 2025 10:00 - 16 Mart 2025 10:00, her gün 08:00-16:00, pist 09/27 kapalı."
    )
    r = ls.score_case("notam", text, ok)
    assert r["unsupported"] == [] and r["recall"] == 1.0, r
    # yanlış gün ve uydurma saat yakalanmalı
    bad = ls.score_case(
        "notam",
        text,
        "LTBA 123: 14 Mart 2025 10:00 - 16 Mart 10:00, 11:00-16:00, pist 09/27 kapalı.",
    )
    assert "from_date" in bad["missing"] and 11.0 in bad["unsupported"]


def test_faithfulness_accepts_coordinate_and_stamp_splits_but_not_wrong_day() -> None:
    from skymind.aviation import llm_summary as ls

    text = (
        "A0123/25 NOTAMN\nQ) LTAA/QWULW/IV/NBO/A/000/999/4059N02845E005\n"
        "A) LTBA B) 2503151000 C) 2503161000\nE) UAS OPR WI 3NM RADIUS 4059N 02845E\nF) SFC G) UNL"
    )
    ok = "LTBA 123: 40°59'K 28°45'D çevresinde 3 NM, 250315 1000 - 250316 1000 arası."
    assert ls.score_case("notam", text, ok)["unsupported"] == []
    # YYAAGG yanlış okunmuş: "25 Mart" (gerçek gün 15)
    bad = ls.score_case("notam", text, "LTBA 123: 25 Mart saat 10:00'dan 26 Mart 10:00'a kadar.")
    assert "from_date" in bad["missing"] and 26.0 in bad["unsupported"]
