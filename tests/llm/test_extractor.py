import pytest

from attrex.confidence.aggregate import Aggregator
from attrex.contracts import Candidate, Card, CategorySchema, Source, Step
from attrex.llm.client import LlmClient
from attrex.llm.extractor import LlmExtractor, LlmSettings
from attrex.normalize.resources import NormalizeResources

CARD = Card(1, "Смартфон Samsung Galaxy S25", "Яркий экран. Ёмкость аккумулятора 4000 мАч. Корпус цвета graphite.")
SCHEMA = CategorySchema(761, ("brand", "battery", "color", "screen_diag"))


@pytest.fixture(scope="module")
def res():
    return NormalizeResources.from_rows([("color", "серый", "серый", 10), ("color", "graphite", "серый", 5),
                                         ("color", "черный", "черный", 10)])


def make(res, transport, samples=3):
    client = LlmClient("m", (0.01, 0.01), budget_rub=10, transport=transport)
    return LlmExtractor(client, res, Aggregator(accept_threshold=0.75), LlmSettings(samples=samples))


def test_targets_are_empty_or_uncertain_with_hook(res, fake):
    ex = make(res, fake())
    prior = [Candidate("brand", "samsung", Step.RULES, Source.TITLE, 0.9, "Samsung", "Samsung"),
             Candidate("color", "черный", Step.RULES, Source.DESCRIPTION, 0.3, "черный", "черный")]
    # brand уверен; battery пуст, но есть зацепка «аккумулятор»; color ниже порога; screen_diag — «экран» в тексте
    assert ex.targets(CARD, SCHEMA, prior) == ["battery", "color", "screen_diag"]
    plain = Card(2, "Чехол", "Мягкий силикон.")
    assert ex.targets(plain, CategorySchema(None, ("battery",)), []) == []     # зацепки нет — LLM не зовём


def test_grounded_majority_vote(res, fake):
    answers = [
        {"battery": {"value": "4000 мАч", "evidence": "аккумулятора 4000 мАч"},
         "color": {"value": "серый", "evidence": "graphite"},
         "screen_diag": {"value": "6.2", "evidence": "экран 6.2 дюйма"}},          # цитаты нет в тексте — выдумка
        {"battery": {"value": "4000", "evidence": "4000 мАч"}, "color": None, "screen_diag": None},
        {"battery": {"value": "5000 мАч", "evidence": "5000 мАч"}, "color": {"value": "серый", "evidence": "graphite"}},
    ]
    t = fake(answers)
    out = {c.attr_id: c for c in make(res, t).extract(CARD, SCHEMA, [])}
    assert t.calls == 3
    assert out["battery"].value == "4000" and out["battery"].score == pytest.approx(0.85 * 2 / 3, abs=1e-4)
    assert out["color"].value == "серый" and out["color"].source is Source.DESCRIPTION
    assert "screen_diag" not in out                                            # не прошло grounding
    assert all(c.step is Step.LLM for c in out.values())


def test_offline_client_does_nothing(res):
    ex = LlmExtractor(LlmClient("m", (0.01, 0.01), 10), res, Aggregator(), LlmSettings())
    assert ex.extract(CARD, SCHEMA, []) == []
    assert ex.drain_usage().llm_calls == 0


def test_prompt_lists_options_and_format(res, fake):
    msgs = make(res, fake()).messages(CARD, ["color", "battery"], [])
    user = msgs[1]["content"]
    color_line = next(line for line in user.splitlines() if line.startswith("- color"))
    assert {"серый", "черный"} <= set(color_line.split("варианты: ")[1].split(", "))
    assert "battery — Ёмкость аккумулятора, число в mAh" in user and "варианты" not in user.split("battery")[1]
    assert '"color": {"value": "...", "evidence": "..."} или null' in user
