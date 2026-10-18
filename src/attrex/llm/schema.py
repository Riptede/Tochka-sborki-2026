"""JSON-схема ответа LLM (b3): схема строится и ответ проверяется здесь, в одном месте.

Схема уходит в запрос как response_format типа json_schema (strict): платформа не даёт модели выйти за неё.
Ответ всё равно проверяется той же схемой: не по схеме — отбрасывается целиком и считается в Usage.llm_invalid.

По каждому атрибуту запроса — объект, все поля обязательны и могут быть null (значения в тексте нет):
    справочный атрибут — value: один из вариантов справочника или null;
    число              — value: число или null, unit: единица, как в тексте;
    текст              — value: строка или null (модель; бренд — справочник слишком велик для перечисления);
    evidence           — дословная цитата из карточки, на которой основано значение.
"""
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

NAME = "attributes"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_NULLABLE_STRING = {"type": ["string", "null"]}


@dataclass(frozen=True)
class AttrField:
    attr_id: str
    kind: Literal["enum", "number", "text"]
    options: tuple[str, ...] = ()

    def schema(self) -> dict:
        if self.kind == "enum":
            value = {"enum": [*self.options, None]}
        elif self.kind == "number":
            value = {"type": ["number", "null"]}
        else:
            value = _NULLABLE_STRING
        props = {"value": value, **({"unit": _NULLABLE_STRING} if self.kind == "number" else {}),
                 "evidence": _NULLABLE_STRING}
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    def raw_value(self, item: dict) -> str | None:
        """Значение ответа строкой для нормализатора b2: число — вместе с единицей."""
        v = item.get("value")
        if v is None:
            return None
        if self.kind != "number":
            return str(v)
        number = int(v) if float(v).is_integer() else v
        return f"{number} {item.get('unit') or ''}".strip()


class AnswerSchema:
    def __init__(self, fields: Sequence[AttrField]):
        self.fields = {f.attr_id: f for f in fields}

    def json_schema(self) -> dict:
        return {"type": "object", "properties": {a: f.schema() for a, f in self.fields.items()},
                "required": list(self.fields), "additionalProperties": False}

    def response_format(self) -> dict:
        return {"type": "json_schema", "json_schema": {"name": NAME, "strict": True, "schema": self.json_schema()}}

    def parse(self, text: str | None) -> dict | None:
        """Ответ модели → объект по схеме; None — не JSON или не по схеме."""
        m = _JSON_BLOCK.search(text or "")
        try:
            answer = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None
        return answer if conforms(answer, self.json_schema()) else None


def _is(value, t: str) -> bool:
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, {"object": dict, "array": list, "string": str, "null": type(None)}[t])


def conforms(value, schema: dict) -> bool:
    """Значение по подмножеству JSON Schema, которое строит этот модуль: type, enum, properties, required,
    additionalProperties."""
    if "enum" in schema and not any(value is e if e is None else type(value) is type(e) and value == e
                                    for e in schema["enum"]):
        return False
    types = schema.get("type")
    if types is not None and not any(_is(value, t) for t in ([types] if isinstance(types, str) else types)):
        return False
    props = schema.get("properties")
    if props is not None and isinstance(value, dict):
        if any(k not in value for k in schema.get("required", ())):
            return False
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return False
        return all(conforms(value[k], s) for k, s in props.items() if k in value)
    return True
