import pytest

from attrex.confidence.aggregate import Aggregator
from attrex.contracts import BuildContext, Candidate, Card, CategorySchema, Source, Status, Step
from attrex.pipeline import Pipeline, SchemaProvider, build_pipeline


class FakeStep:
    """Шаг, который всегда предлагает заданные кандидаты."""

    def __init__(self, step, cands):
        self.step, self.cands = step, cands

    def extract(self, card, schema, prior):
        return [c for c in self.cands if c.attr_id in schema.required]


SCHEMAS = SchemaProvider({761: ("brand", "color", "battery")}, {761: None}, ["brand", "color"],
                         fallback=["brand"], min_conf=0.6)


def test_schema_provider():
    assert SCHEMAS.for_card(Card(1, "x", category_id=761)) == CategorySchema(761, ("brand", "color", "battery"))
    low = SCHEMAS.for_card(Card(1, "x", category_id=761, category_conf=0.4))
    assert low.global_only and low.required == ("brand", "color")      # неуверенная категория — сквозные атрибуты
    unknown = SCHEMAS.for_card(Card(1, "x"))
    assert unknown.global_only and unknown.required == ("brand",)


def make_pipeline(res, *steps):
    return Pipeline(steps, Aggregator(accept_threshold=0.75), res.grounding, SCHEMAS, res.registry)


def test_run_decisions(res):
    card = Card(1, "Смартфон Samsung Galaxy", "Цвет: черный", category_id=761, category_conf=1.0)
    rules = FakeStep(Step.RULES, [
        Candidate("brand", "samsung", Step.RULES, Source.TITLE, 0.9, "Samsung", "Samsung"),
        Candidate("color", "белый", Step.RULES, Source.DESCRIPTION, 0.95, "белый", "белый"),   # evidence не из текста
    ])
    model = FakeStep(Step.MODEL_MATCH, [Candidate("battery", "5000", Step.MODEL_MATCH, Source.OTHER_CARD, 1.0)])
    rows = {p.attr_id: p for p in make_pipeline(res, rules, model).run(card)}
    assert list(rows) == ["brand", "color", "battery"]                 # строка на каждый обязательный атрибут
    assert rows["brand"].status is Status.ACCEPTED and rows["brand"].unit is None
    assert rows["color"].status is Status.REVIEW                       # не прошло grounding
    assert rows["battery"].status is Status.REVIEW and rows["battery"].unit == "mAh"   # model_match — только review


def test_low_confidence_goes_to_review_and_empty_rows(res):
    card = Card(1, "Смартфон Samsung", category_id=761)
    weak = FakeStep(Step.RULES, [Candidate("brand", "samsung", Step.RULES, Source.TITLE, 0.5, "Samsung", "Samsung")])
    rows = {p.attr_id: p for p in make_pipeline(res, weak).run(card)}
    assert rows["brand"].status is Status.REVIEW and rows["brand"].confidence == 0.5
    assert rows["color"].status is Status.EMPTY and rows["color"].step is Step.NONE and rows["color"].value is None


def test_build_pipeline_rejects_unknown_step():
    ctx = BuildContext(con=None, config={"cascade": {"rules": "attrex.normalize.rules"}})
    with pytest.raises(ValueError, match="llm"):
        build_pipeline(ctx, ["rules", "llm"])
