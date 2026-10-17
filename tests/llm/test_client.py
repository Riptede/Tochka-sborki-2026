import pytest

from attrex.llm.client import ApiEncoder, BudgetExceeded, DiskCache, LlmClient, parse_json

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
    llm = {"enabled": True, "budget_rub": 5, "model": "m", "prices_rub_per_1k": {"m": [0.01, 0.02]}}
    client = LlmClient.from_config(llm, {}, tmp_path / "c.sqlite")
    assert not client.available and client.price == (0.01, 0.02)
    with pytest.raises(ValueError, match="нет цены"):
        LlmClient.from_config({**llm, "model": "other"}, {}, None)
