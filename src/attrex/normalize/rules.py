"""Шаг 2 каскада — правила (b2): дёшево и точно по заголовку, осторожнее по описанию.

Каждое правило отвечает за свою группу атрибутов и ничего не знает о других. RulesExtractor применяет те правила,
чьи атрибуты есть в схеме карточки. Новое правило — класс с attr_ids и apply(), добавленный в build().
Все значения правила уже в каноне (ValueNormalizer и units — общие с эталоном), evidence — фрагмент текста.
"""
import re
from collections.abc import Iterable, Iterator
from typing import Protocol

from attrex.contracts import BuildContext, Candidate, Card, CategorySchema, Source, Step
from attrex.normalize.dictionary import DictMatch
from attrex.normalize.dimensions import DimensionLayout, find_dims
from attrex.normalize.numbers import NumberMention, find_numbers, is_cross_unit, unit_fits
from attrex.normalize.resources import NormalizeResources
from attrex.normalize.settings import Scores
from attrex.normalize.text import CardText, TextField
from attrex.normalize.title import TitleLayout
from attrex.normalize.units import canon_model, canon_text, to_canonical_number


class Rule(Protocol):
    attr_ids: frozenset[str]

    def apply(self, text: CardText, wanted: frozenset[str]) -> Iterable[Candidate]: ...


def field_score(scores: Scores, field: TextField) -> float:
    return scores.title if field.source is Source.TITLE else scores.description


def _candidate(attr_id: str, value: str, field: TextField, score: float, start: int, end: int,
               raw: str | None = None) -> Candidate:
    evidence = field.fragment(start, end)
    return Candidate(attr_id=attr_id, value=value, step=Step.RULES, source=field.source, score=round(score, 4),
                     value_raw=evidence if raw is None else raw, evidence=evidence)


class TitleBrandModelRule:
    """Бренд — первое слово-бренд после префикса заголовка, модель — остаток заголовка после бренда."""
    attr_ids = frozenset({"brand", "model"})
    OFFSET_PENALTY = 0.85        # бренд не сразу после префикса — раскладка заголовка менее надёжна
    NO_PREFIX_PENALTY = 0.8      # префикс заголовка не выучен — раскладка угадана
    MODEL_FACTOR = 0.8           # граница модели и характеристик в заголовке размыта сильнее, чем бренд

    def __init__(self, layout: TitleLayout, res: NormalizeResources):
        self.layout, self.res = layout, res

    def apply(self, text: CardText, wanted: frozenset[str]) -> Iterator[Candidate]:
        title, scores = text.title, self.res.settings.scores
        parts = self.layout.parse(title, self.res.dictionary)
        if not parts.found:
            return
        trust = scores.title * self.OFFSET_PENALTY ** parts.offset * (1.0 if parts.prefix_known else self.NO_PREFIX_PENALTY)
        if "brand" in wanted:
            if parts.brand:
                yield _candidate("brand", parts.brand.value, title, trust * parts.brand.score,
                                 parts.brand_start, parts.brand_end)
            else:
                brand = canon_text(title.fragment(parts.brand_start, parts.brand_end))
                if brand:
                    yield _candidate("brand", brand, title, scores.unknown_value, parts.brand_start, parts.brand_end)
        if "model" in wanted:
            model = canon_model(title.text[parts.brand_end:])
            if model:
                yield _candidate("model", model, title, trust * self.MODEL_FACTOR, parts.brand_end, len(title.text))


_MEMORY_PAIR = re.compile(r"(?<![\w.])(\d{1,2})\s*(?:gb|гб)?\s*[/+]\s*(\d{1,4})\s*(gb|гб|tb|тб)(?!\w)")


class MemoryPairRule:
    """«8/256GB», «4GB/128GB», «12+512 ГБ», «16/1TB»: оперативная и встроенная память (у ноутбука — накопитель)."""

    def __init__(self, res: NormalizeResources):
        self.res = res
        self.attr_ids = frozenset({"ram", "storage", "drive"}) & set(res.registry.attrs)

    def apply(self, text: CardText, wanted: frozenset[str]) -> Iterator[Candidate]:
        scores = self.res.settings.scores
        for field in text.fields:
            base = field_score(scores, field)
            for m in _MEMORY_PAIR.finditer(field.folded):
                ram = to_canonical_number(m.group(1), "гб", "GB")
                big = to_canonical_number(m.group(2), m.group(3), "GB")
                if "ram" in wanted and ram:
                    yield _candidate("ram", ram, field, base, m.start(), m.end(), raw=m.group(1))
                if "storage" in wanted and big:
                    yield _candidate("storage", big, field, base, m.start(), m.end(), raw=m.group(2))
                if "drive" in wanted and big:
                    # тип накопителя в такой записи не указан: берём SSD (почти все ноутбуки), значение — на проверку
                    drive = self.res.normalizer.canonical("drive", f"ssd {big} гб")
                    if drive and drive.known:
                        yield _candidate("drive", drive.value, field, base * scores.assumed * drive.score,
                                         m.start(), m.end(), raw=m.group(0))


class UnitNumberRule:
    """Число с единицей → числовой атрибут.

    Если на единицу претендуют несколько атрибутов («ГБ» — память и накопитель, «см» — ширина, высота и глубина),
    выбирает ближайшее ключевое слово из configs/normalize.yaml — среди всех атрибутов реестра, а не только схемы:
    «оперативной памяти 8 ГБ» у смартфона — не встроенная память, даже если RAM в схеме нет. Без ключевого слова
    число берётся только из заголовка и только если единица однозначна для схемы карточки.
    Единицы из anchored_units («т») берутся только с ключевым словом перед числом: «домкрат 3т», «плита 341Т»,
    «1.6Т» у мотора — грузоподъёмность, модель и турбо, а не вес.
    """
    WINDOW_BEFORE = 30     # символов от ключевого слова до числа
    WINDOW_AFTER = 8       # «16GB RAM» — ключевое слово сразу после числа; дальше оно относится к следующему числу
    AFTER_PENALTY = 2      # «Высота 270 см, ширина …»: при равном расстоянии слово перед числом важнее слова после

    def __init__(self, res: NormalizeResources):
        reg, settings = res.registry, res.settings
        self.scores = settings.scores
        self.attr_ids = frozenset(a for a, s in reg.attrs.items() if s.kind == "number")
        # порядок — как в реестре: от него зависит выбор при равных расстояниях, он должен быть одинаков в любом процессе
        self.units = {a: s.unit for a, s in reg.attrs.items() if a in self.attr_ids}
        self.units.update({a: u for a, u in settings.unit_consumers.items() if a in reg.attrs})
        # ключевое слово — целиком («высота», а не «высот»): расстояние до числа считаем от конца слова
        self.keywords = {a: re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, kws)) + r")\w*")
                         for a, kws in settings.keywords.items() if kws}
        self.anchored_units = settings.anchored_units

    def apply(self, text: CardText, wanted: frozenset[str]) -> Iterator[Candidate]:
        if not wanted & self.attr_ids:
            return
        for field in text.fields:
            spans = {a: [m.span() for m in rx.finditer(field.folded)] for a, rx in self.keywords.items()}
            for m in find_numbers(field.folded):
                if m.unit is None or _looks_like_year(m):
                    continue
                attr, anchored = self._resolve(m, wanted, spans, field)
                # число досталось атрибуту вне схемы, атрибуту-справочнику (накопителю) или никому
                if attr not in self.attr_ids or attr not in wanted:
                    continue
                value = m.to(self.units[attr])
                if value is not None:
                    score = field_score(self.scores, field) * (1.0 if anchored else 0.9)
                    yield _candidate(attr, value, field, score, m.start, m.end)

    def _resolve(self, m: NumberMention, wanted: frozenset[str], spans: dict[str, list[tuple[int, int]]],
                 field: TextField) -> tuple[str | None, bool]:
        fits = [a for a, u in self.units.items() if unit_fits(m.unit, u)]
        before_only = m.unit in self.anchored_units
        best, best_d = None, None
        for a in fits:
            for s, e in spans.get(a, ()):
                d = self._distance(field.folded, s, e, m, before_only)
                if d is not None and (best_d is None or d < best_d):
                    best, best_d = a, d
        if best:
            return best, True
        if before_only:
            return None, False
        in_schema = [a for a in fits if a in wanted]
        if field.source is Source.TITLE and len(in_schema) == 1 and not is_cross_unit(m.unit, self.units[in_schema[0]]):
            return in_schema[0], False
        return None, False

    def _distance(self, text: str, s: int, e: int, m: NumberMention, before_only: bool = False) -> int | None:
        if e <= m.start and m.start - e <= self.WINDOW_BEFORE and "\n" not in text[e:m.start]:
            return m.start - e
        if before_only:
            return None
        if s >= m.end and s - m.end <= self.WINDOW_AFTER and "\n" not in text[m.end:s]:
            return (s - m.end) * self.AFTER_PENALTY + 1
        return None


def _looks_like_year(m: NumberMention) -> bool:
    """«2012 г.» — год, а не вес в граммах."""
    return m.unit == "г" and m.number.isdigit() and 1900 <= int(m.number) <= 2100


class DimensionsRule:
    """«100х155», «60x85x60 см» в заголовке → ширина, высота, глубина по выученной раскладке категории."""

    def __init__(self, layout: DimensionLayout, res: NormalizeResources):
        self.layout, self.scores = layout, res.settings.scores
        self.attr_ids = frozenset(a for a, s in res.registry.attrs.items() if s.unit == "cm")

    def apply(self, text: CardText, wanted: frozenset[str]) -> Iterator[Candidate]:
        title = text.title
        dims = find_dims(title.folded)
        if not dims:
            return
        d = dims[0]
        for pos, (attr, rate) in self.layout.layout(text.card.category_id, len(d.values)).items():
            value = d.value_cm(pos)
            if attr in wanted and value:
                yield _candidate(attr, value, title, self.scores.title * rate, d.start, d.end, raw=d.values[pos])


class DictionaryRule:
    """Справочные атрибуты (цвет, материал, процессор, накопитель, состояние) — по написаниям из справочника.

    У атрибутов из title_tail_attrs (цвет) значение из заголовка надёжно, когда стоит в конце: «Смартфон … черный».
    В середине заголовка это чаще часть названия, узора или совместимости («Чехол Red Line … голубой», «орнамент:
    красные»): скор умножается на scores.title_inner. Внутри бренда из справочника («RED SOLUTION») не ищем вовсе.
    """
    MIN_CHARS = 3     # «да», «-» в справочнике есть (шум эталона), но в тексте это не значение атрибута

    def __init__(self, res: NormalizeResources, attr_ids: Iterable[str], layout: TitleLayout | None = None):
        self.res, self.layout = res, layout
        self.attr_ids = frozenset(attr_ids)

    def apply(self, text: CardText, wanted: frozenset[str]) -> Iterator[Candidate]:
        scores = self.res.settings.scores
        tail_attrs = self.res.settings.title_tail_attrs & self.attr_ids & wanted
        brand = self._brand_span(text.title) if tail_attrs else None
        for attr in sorted(self.attr_ids & wanted):
            for field in text.fields:
                positional = attr in tail_attrs and field.source is Source.TITLE
                best: dict[str, tuple[float, DictMatch]] = {}
                for h in self.res.dictionary.find(attr, field.tokens):
                    if h.end - h.start < self.MIN_CHARS:
                        continue
                    score = h.score
                    if positional:
                        if brand and h.start < brand[1] and h.end > brand[0]:
                            continue
                        if not _at_tail(field, h.end):
                            score *= scores.title_inner
                    if h.value not in best or score > best[h.value][0]:
                        best[h.value] = (score, h)
                for score, h in best.values():
                    yield _candidate(attr, h.value, field, field_score(scores, field) * score, h.start, h.end)

    def _brand_span(self, title: TextField) -> tuple[int, int] | None:
        if self.layout is None:
            return None
        parts = self.layout.parse(title, self.res.dictionary)
        return (parts.brand_start, parts.brand_end) if parts.brand else None


def _at_tail(field: TextField, end: int) -> bool:
    """После значения в поле нет слов — только числа, размеры и артикулы («… серый 200х90», «… черный (DP-30HL)»)."""
    return all(any(ch.isdigit() for ch in t.text) for t in field.tokens if t.start >= end)


class RulesExtractor:
    step = Step.RULES

    def __init__(self, rules: Iterable[Rule]):
        self.rules = tuple(rules)

    def extract(self, card: Card, schema: CategorySchema, prior: list[Candidate]) -> list[Candidate]:
        wanted = frozenset(schema.required)
        text = CardText.of(card)
        return [c for rule in self.rules if rule.attr_ids & wanted for c in rule.apply(text, wanted)]


def default_rules(res: NormalizeResources, layout: TitleLayout, dims: DimensionLayout) -> list[Rule]:
    """Набор правил базлайна. Справочные атрибуты, кроме бренда (его даёт раскладка заголовка), — по справочнику."""
    dictionary_attrs = [a for a, s in res.registry.attrs.items() if s.kind == "category" and a != "brand"]
    return [
        TitleBrandModelRule(layout, res),
        MemoryPairRule(res),
        UnitNumberRule(res),
        DimensionsRule(dims, res),
        DictionaryRule(res, dictionary_attrs, layout),
    ]


def build(ctx: BuildContext) -> RulesExtractor:
    res = ctx.resource(NormalizeResources.from_context)
    return RulesExtractor(default_rules(res, ctx.resource(TitleLayout.from_context),
                                        ctx.resource(DimensionLayout.from_context)))
