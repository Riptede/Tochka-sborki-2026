import json

import pytest

from attrex.llm.schema import AnswerSchema, AttrField, conforms

SCHEMA = AnswerSchema([AttrField("color", "enum", ("серый", "черный")), AttrField("battery", "number"),
                       AttrField("model", "text")])
OK = {"color": {"value": "серый", "evidence": "graphite"},
      "battery": {"value": 4000, "unit": "мАч", "evidence": "4000 мАч"},
      "model": {"value": None, "evidence": None}}


def test_json_schema_enum_number_text():
    s = SCHEMA.response_format()
    assert s["type"] == "json_schema" and s["json_schema"]["strict"] is True
    root = s["json_schema"]["schema"]
    assert root["required"] == ["color", "battery", "model"] and root["additionalProperties"] is False
    color, battery, model = (root["properties"][a] for a in ("color", "battery", "model"))
    assert color["properties"]["value"] == {"enum": ["серый", "черный", None]}
    assert battery["required"] == ["value", "unit", "evidence"]
    assert battery["properties"]["value"] == {"type": ["number", "null"]}
    assert model["properties"]["value"] == {"type": ["string", "null"]} and "unit" not in model["properties"]
    assert conforms(OK, root)


@pytest.mark.parametrize("broken", [
    {**OK, "color": {"value": "зеленый", "evidence": "зеленый"}},              # не из вариантов справочника
    {**OK, "battery": {"value": "4000", "unit": "мАч", "evidence": "4000"}},    # число строкой
    {**OK, "battery": {"value": True, "unit": None, "evidence": None}},        # bool — не число
    {k: v for k, v in OK.items() if k != "model"},                             # атрибута нет в ответе
    {**OK, "ram": {"value": 8, "unit": "GB", "evidence": "8 ГБ"}},             # лишний атрибут
    {**OK, "color": {"value": "серый"}},                                       # нет evidence
    {**OK, "color": None},                                                     # объект атрибута обязателен
])
def test_answer_not_matching_schema_is_rejected(broken):
    assert SCHEMA.parse(json.dumps(broken, ensure_ascii=False)) is None


def test_parse_valid_and_wrapped():
    assert SCHEMA.parse(json.dumps(OK, ensure_ascii=False)) == OK
    assert SCHEMA.parse("```json\n" + json.dumps(OK, ensure_ascii=False) + "\n```") == OK
    assert SCHEMA.parse("не json") is None and SCHEMA.parse("") is None and SCHEMA.parse("[1, 2]") is None


def test_raw_value_for_normalizer():
    battery, color = SCHEMA.fields["battery"], SCHEMA.fields["color"]
    assert battery.raw_value({"value": 4000.0, "unit": "мАч"}) == "4000 мАч"
    assert battery.raw_value({"value": 6.7, "unit": None}) == "6.7"
    assert color.raw_value({"value": "серый"}) == "серый"
    assert color.raw_value({"value": None}) is None
