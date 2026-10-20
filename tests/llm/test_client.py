import pytest

from attrex.llm.client import (
    MAX_CONSECUTIVE_FAILURES,
    ApiEncoder,
    BudgetExceeded,
    DiskCache,
    LlmClient,
    PermanentApiError,
    parse_json,
)
from attrex.llm.schema import AnswerSchema, AttrField

PRICE = (0.0122, 0.0122)
MSG = [{"role": "user", "content": "Цвет?"}]


def test_parse_json_tolerates_wrapping():
    assert parse_json('Вот ответ: ```json\n{"color": null}\n```') == {"color": None}
    assert parse_json("нет json") is None
    assert parse_json("[1, 2]") is None


def test_no_transport_means_no_calls():
    client = LlmClient("m", PRICE, budget_rub=10)
    assert not client.available and client.chat_json(MSG) is None


def test_zero_budget_forbids_calls(fake):
    t = fake([{"color": None}])
    client = LlmClient("m", PRICE, budget_rub=0, transport=t)
    assert client.chat_json(MSG) is None and t.calls == 0


def test_cache_usage_and_cost(tmp_path, fake):
    t = fake([{"color": {"value": "черный", "evidence": "черный"}}], prompt_tokens=1000, completion_tokens=500)
    client = LlmClient("m", PRICE, budget_rub=1, cache=DiskCache(tmp_path / "c.sqlite"), transport=t)
    first = client.chat_json(MSG)
    again = client.chat_json(MSG)
    assert first == again and t.calls == 1                       # второй раз — из кэша, без оплаты
    usage = client.drain_usage()
    assert (usage.llm_calls, usage.llm_tokens) == (1, 1500)
    assert usage.llm_cost_rub == pytest.approx(1.5 * 0.0122)
    assert client.drain_usage().llm_calls == 0                    # drain обнуляет счётчик


def test_budget_checked_before_call(fake):
    t = fake([{"x": 1}], prompt_tokens=1000, completion_tokens=600)
    client = LlmClient("m", (1.0, 1.0), budget_rub=1.0, transport=t, max_tokens=600)
    assert client.chat_json(MSG) is not None                      # худший случай ≈ 0.6 ₽ — влезает
    assert client.chat_json(MSG + [{"role": "user", "content": "ещё"}]) is None   # потрачено 1.6 ₽ — больше нет
    assert t.calls == 1 and client.skipped_by_budget == 1


def test_embeddings_via_client(tmp_path, fake):
    t = fake()
    client = LlmClient("m", PRICE, 0, cache=DiskCache(tmp_path / "c.sqlite"), transport=t,
                       embed_model="bge", embed_price=0.00061, embed_budget_rub=1)
    enc = ApiEncoder(client)
    v = enc.encode(["черный", "белый"])
    assert v.shape == (2, 4) and abs(float(v[0] @ v[0]) - 1) < 1e-6
    enc.encode(["черный"])
    assert t.embed_calls == 1                                     # уже посчитанное — из кэша
    assert enc.drain_usage().embed_tokens == 20
    with pytest.raises(BudgetExceeded):
        ApiEncoder(LlmClient("m", PRICE, 0)).encode(["серый"])


def test_from_config_without_key_is_offline(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = {"enabled": True, "budget_rub": 5, "model": "m", "prices_rub_per_1k": {"m": [0.01, 0.02]}, "max_retries": 4}
    client = LlmClient.from_config(llm, {}, tmp_path / "c.sqlite")
    assert not client.available and client.price == (0.01, 0.02) and client.max_retries == 4
    with pytest.raises(ValueError, match="нет цены"):
        LlmClient.from_config({**llm, "model": "other"}, {}, None)


def failing(fake, **kw):
    return LlmClient("m", PRICE, budget_rub=10, transport=fake(**kw), max_retries=2, retry_backoff_s=0)


def test_api_error_gives_none_and_failure_count(fake):
    client = failing(fake, fail_always=TimeoutError("read timeout"))
    assert client.chat_json(MSG) is None                          # исключение не вылетает из клиента
    assert client.transport.calls == 3                            # 1 попытка + max_retries повторов
    usage = client.drain_usage()
    assert (usage.api_failures, usage.llm_calls, usage.llm_cost_rub) == (1, 0, 0)


def test_retry_then_success(fake):
    client = failing(fake, answers=[{"color": None}], errors=[ConnectionError("reset")])
    assert client.chat_json(MSG) == {"color": None}
    usage = client.drain_usage()
    assert client.transport.calls == 2 and usage.api_failures == 0 and usage.llm_calls == 1


def test_permanent_error_is_not_retried(fake):
    client = failing(fake, fail_always=PermanentApiError("BadRequestError 400"))
    assert client.chat_json(MSG) is None and client.transport.calls == 1


def test_many_failures_in_a_row_stop_calls(fake):
    client = failing(fake, fail_always=TimeoutError())
    for i in range(MAX_CONSECUTIVE_FAILURES + 5):
        assert client.chat_json([{"role": "user", "content": f"вопрос {i}"}]) is None
    assert client.transport.calls == 3 * MAX_CONSECUTIVE_FAILURES   # дальше API не зовём
    usage = client.drain_usage()
    assert usage.api_failures == MAX_CONSECUTIVE_FAILURES and usage.llm_skipped == 5


def test_embed_error_gives_none(fake, tmp_path):
    t = fake(fail_always=OSError("network is unreachable"))
    client = LlmClient("m", PRICE, 0, transport=t, embed_model="bge", embed_price=0.00061, embed_budget_rub=1,
                       max_retries=1, retry_backoff_s=0)
    assert client.embed(["черный"]) is None and t.embed_calls == 2
    assert client.drain_usage().api_failures == 1
    with pytest.raises(BudgetExceeded):                           # retrieval ловит его и идёт дальше
        ApiEncoder(client).encode(["черный"])


def test_schema_goes_to_request_and_checks_answer(fake, tmp_path):
    schema = AnswerSchema([AttrField("color", "enum", ("серый", "черный"))])
    good = {"color": {"value": "серый", "evidence": "graphite"}}
    t = fake([good, {"color": {"value": "зеленый", "evidence": "green"}}])
    client = LlmClient("m", PRICE, budget_rub=10, cache=DiskCache(tmp_path / "c.sqlite"), transport=t)
    assert client.chat_json(MSG, schema=schema) == good
    assert t.extras[0]["response_format"] == schema.response_format()
    assert client.chat_json(MSG, seed=1, schema=schema) is None   # «зеленый» нет в вариантах — ответ отброшен
    assert client.drain_usage().llm_invalid == 1
    assert client.chat_json(MSG, schema=schema) == good and t.calls == 2   # из кэша: схема — часть ключа
