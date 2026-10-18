import pytest

from attrex.confidence.aggregate import Aggregator
from attrex.contracts import Candidate, Card, CategorySchema, Source, Step
from attrex.llm.client import LlmClient
from attrex.llm.extractor import LlmExtractor, LlmSettings
from attrex.normalize.resources import NormalizeResources

CARD = Card(1, "Смартфон Samsung Galaxy S25", "Яркий экран. Ёмкость аккумулятора 4000 мАч. Корпус цвета graphite.")
SCHEMA = CategorySchema(761, ("brand", "battery", "color", "screen_diag"))
BRAND = Candidate("brand", "samsung", Step.RULES, Source.TITLE, 0.9, "Samsung", "Samsung")   # уверен: не цель LLM
NULL_NUM = {"value": None, "unit": None, "evidence": None}
NULL = {"value": None, "evidence": None}


@pytest.fixture(scope="module")
def res():
    return NormalizeResources.from_rows([("color", "серый", "серый", 10), ("color", "graphite", "серый", 5),
                                         ("color", "черный", "черный", 10)])


def make(res, transport, samples=3):
    client = LlmClient("m", (0.01, 0.01), budget_rub=10, transport=transport, retry_backoff_s=0)
    return LlmExtractor(client, res, Aggregator(accept_threshold=0.75), LlmSettings(samples=samples))


def test_targets_are_empty_or_uncertain_with_hook(res, fake):
    ex = make(res, fake())
    prior = [BRAND, Candidate("color", "черный", Step.RULES, Source.DESCRIPTION, 0.3, "черный", "черный")]
    # brand уверен; battery пуст, но есть зацепка «аккумулятор»; color ниже порога; screen_diag — «экран» в тексте
    assert ex.targets(CARD, SCHEMA, prior) == ["battery", "color", "screen_diag"]
    plain = Card(2, "Чехол", "Мягкий силикон.")
    assert ex.targets(plain, CategorySchema(None, ("battery",)), []) == []     # зацепки нет — LLM не зовём


def test_grounded_majority_vote(res, fake):
    answers = [
        {"battery": {"value": 4000, "unit": "мАч", "evidence": "аккумулятора 4000 мАч"},
         "color": {"value": "серый", "evidence": "graphite"},
         "screen_diag": {"value": 6.2, "unit": "дюйма", "evidence": "экран 6.2 дюйма"}},   # цитаты нет в тексте
        {"battery": {"value": 4000, "unit": None, "evidence": "4000 мАч"}, "color": NULL, "screen_diag": NULL_NUM},
        {"battery": {"value": 5000, "unit": "мАч", "evidence": "5000 мАч"},                 # выдумка
         "color": {"value": "серый", "evidence": "graphite"}, "screen_diag": NULL_NUM},
    ]
    t = fake(answers)
    ex = make(res, t)
    out = {c.attr_id: c for c in ex.extract(CARD, SCHEMA, [BRAND])}
    assert t.calls == 3
    assert out["battery"].value == "4000" and out["battery"].score == pytest.approx(0.85 * 2 / 3, abs=1e-4)
    assert out["color"].value == "серый" and out["color"].source is Source.DESCRIPTION
    assert "screen_diag" not in out                                            # не прошло grounding
    assert all(c.step is Step.LLM for c in out.values())
    usage = ex.drain_usage()
    assert (usage.llm_cards, usage.llm_calls, usage.llm_values, usage.llm_ungrounded) == (1, 3, 6, 2)


def test_request_carries_schema_with_dictionary_options(res, fake):
    t = fake([{"battery": NULL_NUM, "color": NULL, "screen_diag": NULL_NUM}])
    make(res, t, samples=1).extract(CARD, SCHEMA, [BRAND])
    props = t.extras[0]["response_format"]["json_schema"]["schema"]["properties"]
    assert set(props) == {"battery", "color", "screen_diag"}
    assert {"серый", "черный", None} <= set(props["color"]["properties"]["value"]["enum"])
    assert props["battery"]["properties"]["value"] == {"type": ["number", "null"]}
    assert t.extras[0]["response_format"]["json_schema"]["strict"] is True


def test_answer_not_by_schema_is_dropped_and_counted(res, fake):
    good = {"battery": {"value": 4000, "unit": "мАч", "evidence": "4000 мАч"}, "color": NULL, "screen_diag": NULL_NUM}
    off_list = {**good, "color": {"value": "графит в крапинку", "evidence": "graphite"}}   # нет среди вариантов
    ex = make(res, fake([good, off_list, "не json"]))
    out = ex.extract(CARD, SCHEMA, [BRAND])
    assert [(c.attr_id, c.value) for c in out] == [("battery", "4000")]
    assert out[0].score == pytest.approx(0.85 / 3, abs=1e-4)                  # голосует один ответ из трёх
    assert ex.drain_usage().llm_invalid == 2


def test_api_failure_does_not_stop_the_run(res, fake):
    t = fake(fail_always=ConnectionError("connection reset by peer"))
    ex = make(res, t)
    assert ex.extract(CARD, SCHEMA, [BRAND]) == []                             # исключение не вылетает
    usage = ex.drain_usage()
    assert usage.api_failures == 3 and usage.llm_calls == 0 and usage.llm_cards == 1
    assert t.calls == 9                                                        # 3 ответа × (1 + 2 повтора)


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
    assert "value и evidence" in user and "null" in user
