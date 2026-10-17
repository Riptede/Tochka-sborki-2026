"""Сырое значение из текста → канон атрибута. Одно правило для всех шагов: правила, парсер, LLM и grounding.

Канон тот же, что у эталона core_gold (b1): числа — `units.to_canonical_number`, модель — `units.canon_model`,
справочные атрибуты — значение справочника, остальной текст — `units.canon_text`.
"""
from dataclasses import dataclass

from attrex.normalize.dictionary import RefDictionary
from attrex.normalize.keys import KEY_FUNCTIONS
from attrex.normalize.numbers import numbers_in, same_number
from attrex.normalize.settings import Scores
from attrex.normalize.text import fold, phrase_key, tokenize
from attrex.normalize.units import canon_model, canon_text, to_canonical_number
from attrex.registry import Registry


@dataclass(frozen=True)
class CanonValue:
    value: str
    score: float      # насколько надёжен перевод в канон: доля голосов справочника; 1 для чисел
    known: bool       # значение есть в справочнике (для чисел и текста — всегда True)


def canon_string(attr_id: str, s: str | None) -> str | None:
    """Канон текстового значения — как у эталона: модель без памяти, цвета и артикулов, остальное — canon_text."""
    return canon_model(s) if attr_id == "model" else canon_text(s)


class ValueNormalizer:
    def __init__(self, registry: Registry, dictionary: RefDictionary, scores: Scores,
                 key_functions: dict[str, str] | None = None):
        self.registry = registry
        self.dictionary = dictionary
        self.scores = scores
        self._keys = {a: KEY_FUNCTIONS[name] for a, name in (key_functions or {}).items()}

    def canonical(self, attr_id: str, raw: str | None, unit: str | None = None) -> CanonValue | None:
        """Канон сырого значения или None, если это заглушка или значение не переводится."""
        if raw is None or self.registry.is_placeholder(raw):
            return None
        spec = self.registry[attr_id]
        if spec.kind == "number":
            v = to_canonical_number(raw, unit, spec.unit)
            return CanonValue(v, 1.0, True) if v is not None else None
        if spec.kind == "text":
            v = canon_string(attr_id, raw)
            return CanonValue(v, 1.0, True) if v else None
        return self._category(attr_id, raw)

    def _category(self, attr_id: str, raw: str) -> CanonValue | None:
        hit = self.dictionary.lookup(attr_id, raw)
        if hit:
            return CanonValue(hit.value, hit.score, True)
        inner = self.dictionary.find(attr_id, tokenize(fold(raw)))
        if inner:
            best = max(inner, key=lambda h: (h.end - h.start, h.support))
            return CanonValue(best.value, best.score * self.scores.partial_match, True)
        v = canon_string(attr_id, raw)
        return CanonValue(v, self.scores.unknown_value, False) if v else None

    def same(self, attr_id: str, a: str | None, b: str | None) -> bool:
        """Два канона — одно значение: числа с точностью округления, процессор — по модели без частоты
        (частоты в эталоне у одной модели расходятся, а в тексте их обычно нет; EDA, раздел 7)."""
        if a is None or b is None:
            return False
        if self.registry[attr_id].kind == "number":
            return same_number(a, b)
        key_fn = self._keys.get(attr_id)
        if key_fn and key_fn(a) is not None and key_fn(a) == key_fn(b):
            return True
        return a == b

    def supports(self, attr_id: str, fragment: str, value: str) -> bool:
        """Фрагмент текста подтверждает значение: сравнение после нормализации, а не по строке."""
        if not fragment or not value:
            return False
        spec = self.registry[attr_id]
        if spec.kind == "number":
            return any(same_number(v, value) for v in numbers_in(fold(fragment), spec.unit))
        if spec.kind == "category":
            key_fn = self._keys.get(attr_id)
            if key_fn and key_fn(fragment) is not None and key_fn(fragment) == key_fn(value):
                return True
            if any(h.value == value for h in self.dictionary.find(attr_id, tokenize(fold(fragment)))):
                return True
        return _contains_words(phrase_key(fragment), phrase_key(value))


def _contains_words(text_key: str, value_key: str) -> bool:
    """Все слова значения есть во фрагменте; порядок слов не важен, части через дефис — тоже слова
    («темно-серый» подтверждает «серый»)."""
    words = set(text_key.split())
    words |= {part for w in words if "-" in w for part in w.split("-")}
    return bool(value_key) and all(w in words for w in value_key.split())
