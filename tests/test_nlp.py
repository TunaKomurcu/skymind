"""NER hizalama, BIO/varlık F1, elle yazılmış set okuyucusu, niyet etiketi."""

import pytest

from skymind.nlp.datasets import manual_ner, parse_marked
from skymind.nlp.metrics import bio_spans, entity_f1
from skymind.nlp.ner_data import align_example, intent_of, rule_tagger, words


def test_bio_spans_follow_conlleval_rules() -> None:
    assert bio_spans(["O", "B-ALT", "I-ALT", "O", "B-DIR"]) == {("ALT", 1, 2), ("DIR", 4, 4)}
    # B'siz I yeni varlık başlatır; tür değişince önceki kapanır
    assert bio_spans(["I-ALT", "I-ALT"]) == {("ALT", 0, 1)}
    assert bio_spans(["B-ALT", "I-DIR"]) == {("ALT", 0, 0), ("DIR", 1, 1)}
    assert bio_spans(["B-ALT", "B-ALT"]) == {("ALT", 0, 0), ("ALT", 1, 1)}
    assert bio_spans([]) == set()


def test_entity_f1_exact_match_and_partial_credit_is_zero() -> None:
    gold = [["B-ALT", "I-ALT", "O"], ["B-DIR", "O", "O"]]
    perfect = entity_f1(gold, gold)
    assert perfect["micro"]["f1"] == 1.0 and perfect["ALT"]["f1"] == 1.0
    # sınır hatası (yalnızca ilk kelime): varlık eşleşmez, hem FP hem FN
    partial = entity_f1(gold, [["B-ALT", "O", "O"], ["B-DIR", "O", "O"]])
    assert partial["ALT"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert partial["micro"]["precision"] == 0.5 and partial["micro"]["recall"] == 0.5
    with pytest.raises(ValueError):
        entity_f1(gold, gold[:1])
    with pytest.raises(ValueError):
        entity_f1([["O"]], [["O", "O"]])


PLAN = {
    "type": "plan",
    "steps": [
        {"action": "takeoff", "altitude_m": 1500.0},
        {
            "action": "scan",
            "center": {"kind": "named", "name": "baraj"},
            "radius_m": 3000.0,
            "altitude_m": 500.0,
        },
        {"action": "goto", "target": {"kind": "relative", "direction": "N", "distance_m": 2000.0}},
    ],
}


def test_alignment_labels_values_by_context_and_value() -> None:
    cmd = "1,5 km irtifaya çık, barajı 3 km yarıçapta 500 metre yükseklikte tara, kuzeye 2 km git"
    al = align_example(cmd, PLAN)
    assert al is not None
    ws, tags = al
    got = {(w, t) for w, t in zip(ws, tags, strict=True) if t != "O"}
    assert ("1,5", "B-ALT") in got and ("km", "I-ALT") in got
    assert ("barajı", "B-PLACE") in got
    assert ("3", "B-RADIUS") in got and ("500", "B-ALT") in got
    assert ("kuzeye", "B-DIR") in got and ("2", "B-DIST") in got


def test_alignment_rejects_when_a_gold_value_is_missing_from_text() -> None:
    # altın plan 1500 m irtifa içerir ama komutta yok -> hizalanamaz (zayıf etiketi uydurmaz)
    assert (
        align_example("barajı 3 km yarıçapta 500 metre yükseklikte tara, kuzeye 2 km git", PLAN)
        is None
    )


def test_alignment_handles_attached_units_and_repeated_places() -> None:
    plan = {
        "type": "plan",
        "steps": [
            {"action": "goto", "target": {"kind": "named", "name": "depo"}},
            {"action": "land", "at": {"kind": "named", "name": "depo"}},
            {"action": "goto", "target": {"kind": "relative", "direction": "E", "distance_m": 500}},
        ],
    }
    al = align_example("depoya git, depoya in, doğuya 500m git", plan)
    assert al is not None
    _, tags = al
    assert tags.count("B-PLACE") == 2 and "B-DIST" in tags


def test_rule_tagger_does_not_need_gold_and_defaults_unknown_lengths() -> None:
    tags = rule_tagger("500 metre irtifaya çık, kuzeye 1 km git")
    assert "B-ALT" in tags and "B-DIR" in tags and "B-DIST" in tags


def test_words_strip_edge_punctuation_but_keep_inner() -> None:
    assert [w[0] for w in words("Hızı 1,5 m/s yap, lütfen.")] == [
        "Hızı",
        "1,5",
        "m/s",
        "yap",
        "lütfen",
    ]


def test_intent_priority_and_non_plan_types() -> None:
    assert intent_of(PLAN) == "scan"
    assert (
        intent_of({"type": "plan", "steps": [{"action": "takeoff", "altitude_m": 100}]})
        == "takeoff"
    )
    assert intent_of({"type": "clarification"}) == "clarification"
    assert intent_of({"type": "refusal"}) == "refusal"


def test_parse_marked_builds_bio_over_words() -> None:
    w, t = parse_marked("[500 m|ALT] irtifada [Alfa noktasına|PLACE] git.")
    assert w == ["500", "m", "irtifada", "Alfa", "noktasına", "git"]
    assert t == ["B-ALT", "I-ALT", "O", "B-PLACE", "I-PLACE", "O"]
    for bad in ("[x|YOK] a", "[a b", "düz |bar"):
        with pytest.raises(ValueError):
            parse_marked(bad)


def test_manual_set_parses_and_covers_every_entity_type() -> None:
    sets = manual_ner()
    assert set(sets) == {"KOLAY", "ZOR"} and all(len(v[0]) >= 20 for v in sets.values())
    types = {t[2:] for _, ts in sets.values() for tags in ts for t in tags if t != "O"}
    assert types == {"ALT", "RADIUS", "DIST", "SPEED", "DUR", "COORD", "PLACE", "DIR"}
    for ws, ts in sets.values():
        assert all(len(w) == len(t) for w, t in zip(ws, ts, strict=True))
