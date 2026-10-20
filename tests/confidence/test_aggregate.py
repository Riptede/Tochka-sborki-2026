import pytest

from attrex.confidence.aggregate import Aggregator
from attrex.contracts import Candidate, Source, Step
from attrex.normalize.numbers import same_number


def cand(attr, value, score, step=Step.RULES, source=Source.TITLE):
    return Candidate(attr, value, step, source, score, evidence=value)


def same(attr_id, a, b):
    return same_number(a, b) if attr_id == "storage" else a == b


@pytest.fixture
def agg():
    return Aggregator({"rules": 1.0, "parse": 1.0, "model_match": 0.5}, same, accept_threshold=0.75,
                      thresholds={"color": 0.9})


def test_independent_sources_increase_confidence(agg):
    one = agg.decide([cand("color", "черный", 0.6)], ["color"])["color"]
    two = agg.decide([cand("color", "черный", 0.6),
                      cand("color", "черный", 0.6, Step.PARSE, Source.SELLER_ATTRS)], ["color"])["color"]
    assert one.value == two.value == "черный"
    assert one.confidence == 0.6
    assert two.confidence == pytest.approx(0.84)            # 1 − 0.4 · 0.4
    assert len(two.support) == 2


def test_same_step_and_source_is_one_confirmation(agg):
    d = agg.decide([cand("storage", "256", 0.9), cand("storage", "256.0", 0.8)], ["storage"])["storage"]
    assert d.confidence == 0.9 and d.best.score == 0.9


def test_conflict_lowers_confidence(agg):
    d = agg.decide([cand("color", "черный", 0.9), cand("color", "белый", 0.6, source=Source.DESCRIPTION)],
                   ["color"])["color"]
    assert d.value == "черный" and d.n_values == 2
    assert d.confidence == pytest.approx(0.9 * 0.9 / 1.5)


def test_empty_attr_and_step_weights(agg):
    out = agg.decide([cand("battery", "5000", 0.8, Step.MODEL_MATCH, Source.OTHER_CARD)], ["battery", "color"])
    assert out["color"].empty and out["color"].confidence == 0.0
    assert out["battery"].confidence == pytest.approx(0.4)    # model_match весит вдвое меньше


def test_thresholds(agg):
    assert agg.threshold("color") == 0.9
    assert agg.threshold("brand") == 0.75


def test_calibrator_is_applied_with_step():
    agg = Aggregator(calibrator=lambda attr_id, step, raw: raw / 2 if step is Step.RULES else raw)
    assert agg.decide([cand("color", "черный", 0.8)], ["color"])["color"].confidence == 0.4
    parse = cand("color", "черный", 0.8, Step.PARSE, Source.SELLER_ATTRS)
    assert agg.decide([parse], ["color"])["color"].confidence == 0.8


def test_llm_confirmation_uses_llm_curve():
    agg = Aggregator(calibrator=lambda attr_id, step, raw: 0.99 if step is Step.LLM else 0.5)
    rules = cand("color", "черный", 0.9)
    assert agg.decide([rules], ["color"])["color"].confidence == 0.5
    confirmed = agg.decide([rules, cand("color", "черный", 0.7, Step.LLM)], ["color"])["color"]
    assert confirmed.confidence == 0.99 and confirmed.best.step is Step.RULES   # кривая LLM, цитата — сильнейшая
    other = agg.decide([rules, cand("color", "белый", 0.7, Step.LLM)], ["color"])["color"]
    assert other.value == "черный" and other.confidence == 0.5                  # LLM за другое значение — не подтверждение
