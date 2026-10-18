"""Где в тексте карточки лежит значение эталона: core_gold.in_text (EDA, раздел 7).

    title        значение есть в заголовке (слова — с точностью до окончания, числа — с переводом единиц)
    description  значение есть только в описании
    synonym      есть другое написание этого значения: «titanium grey» для «серый»
    partial      есть не меньше половины слов значения: «Intel Core i5 13420H» без частоты «2.1 ГГц»
    none         в тексте значения нет — честно извлечь его из этой карточки нельзя

Нужно для метрик: полнота на подмножестве, где значение есть в тексте, отделяет качество извлечения от того,
что извлечь в принципе нельзя. Сравнение — общими функциями нормализатора (`normalize.text`, `normalize.numbers`),
теми же, что у grounding, а не своими правилами.
"""
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass

from attrex.normalize.numbers import find_numbers, same_number, unit_fits
from attrex.normalize.text import fold, stem, tokenize

LEVELS = ("title", "description", "synonym", "partial", "none")
_UNIT_WORDS = {"гб", "тб", "мб", "gb", "tb", "мач", "mah", "ггц", "ghz", "см", "мм", "г", "кг", "in"}


def stems(text: str) -> str:
    """Основы слов через пробел и с пробелами по краям: поиск фразы — поиск подстроки « основа основа »."""
    return " " + " ".join(stem(t.text) for t in tokenize(fold(text))) + " "


def _parts(s: str) -> list[str]:
    """Основы слов и части слов через дефис: «i5-13420h» → «i5-13420h», «i5», «13420h»."""
    out = []
    for w in s.split():
        out.append(w)
        if "-" in w:
            out += [p for p in w.split("-") if p]
    return out


def phrase_tokens(value: str) -> list[str]:
    """Значимые слова значения для частичного совпадения: без единиц и однобуквенных слов."""
    return [s for s in dict.fromkeys(_parts(stems(value))) if s not in _UNIT_WORDS and (len(s) >= 2 or s.isdigit())]


@dataclass
class CardText:
    title: str
    description: str

    def __post_init__(self):
        self.title_f, self.desc_f = fold(self.title or ""), fold(self.description or "")
        self.title_s, self.desc_s = stems(self.title or ""), stems(self.description or "")
        self.words = set(_parts(self.title_s)) | set(_parts(self.desc_s))
        self._numbers: dict[str, list[tuple[str, str]]] = {}

    def numbers(self, unit: str) -> list[tuple[str, str]]:
        """(где, число в единице unit) для всех чисел текста; считается один раз на единицу."""
        if unit not in self._numbers:
            out = []
            for place, folded in (("title", self.title_f), ("description", self.desc_f)):
                for m in find_numbers(folded):
                    if m.unit is None and len(m.number.replace(".", "")) < 2:
                        continue
                    if m.unit is not None and not unit_fits(m.unit, unit):
                        continue
                    x = m.to(unit)
                    if x is not None:
                        out.append((place, x))
            self._numbers[unit] = out
        return self._numbers[unit]


def locate_number(value: str, unit: str, card: CardText) -> str:
    """Число эталона среди чисел текста, переведённых в его единицу. Однозначное число без единицы не считаем:
    в тексте их слишком много («8/256», «2 в 1»)."""
    for place, x in card.numbers(unit):
        if same_number(x, value):
            return place
    return "none"


@dataclass(frozen=True)
class Phrase:
    """Значение справочника, подготовленное к поиску: основы фразы, синонимов и отдельных слов."""
    key: str
    synonyms: tuple[str, ...]
    tokens: tuple[str, ...]

    @classmethod
    def of(cls, value: str, synonyms: Sequence[str] = ()) -> "Phrase":
        syn = tuple(k for k in dict.fromkeys(stems(s) for s in synonyms) if k.strip())
        return cls(stems(value), syn, tuple(phrase_tokens(value)))


def locate_phrase(p: Phrase, card: CardText) -> str:
    if p.key.strip():
        if p.key in card.title_s:
            return "title"
        if p.key in card.desc_s:
            return "description"
    for k in p.synonyms:
        if k in card.title_s or k in card.desc_s:
            return "synonym"
    toks = p.tokens
    if toks:
        hit = sum(t in card.words for t in toks)
        if hit and hit / len(toks) >= 0.5:
            return "partial"
    return "none"


class Locator:
    """in_text для строк эталона. kinds: attr_id → (kind, unit); synonyms: (attr_id, value) → написания."""

    def __init__(self, kinds: Mapping[str, tuple[str, str | None]], synonyms: Mapping[tuple[str, str], Sequence[str]]):
        self.kinds, self.synonyms = kinds, synonyms
        self._phrases: dict[tuple[str, str], Phrase] = {}

    def phrase(self, attr_id: str, value: str) -> Phrase:
        key = (attr_id, value)
        if key not in self._phrases:
            self._phrases[key] = Phrase.of(value, self.synonyms.get(key, ()))
        return self._phrases[key]

    def locate(self, attr_id: str, value: str, card: CardText) -> str:
        kind, unit = self.kinds[attr_id]
        if kind == "number":
            return locate_number(value, unit, card)
        return locate_phrase(self.phrase(attr_id, value), card)

    def locate_many(self, cards: Mapping[int, CardText], rows: Iterable[tuple[int, str, str]]
                    ) -> list[tuple[int, str, str, str]]:
        empty = CardText("", "")
        return [(sku, a, v, self.locate(a, v, cards.get(sku, empty))) for sku, a, v in rows]


Chunk = tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]   # (карточки: sku, title, description; строки эталона)
_worker: Locator | None = None


def _init_worker(locator: Locator) -> None:
    global _worker
    _worker = locator


def _locate_chunk(chunk: Chunk) -> list[tuple[int, str, str, str]]:
    cards, rows = chunk
    return _worker.locate_many({s: CardText(t, d) for s, t, d in cards}, rows)


def locate_chunks(locator: Locator, chunks: Iterable[Chunk], workers: int = 1) -> Iterator[list[tuple[int, str, str, str]]]:
    """in_text по пачкам карточек; workers > 1 — в процессах. В очереди не больше 2·workers пачек."""
    if workers <= 1:
        _init_worker(locator)
        yield from map(_locate_chunk, chunks)
        return
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(locator,)) as pool:
        pending: deque[Future] = deque()
        for chunk in chunks:
            pending.append(pool.submit(_locate_chunk, chunk))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()