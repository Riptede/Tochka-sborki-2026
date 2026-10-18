"""Шаг 3 каскада — парсер блоков «Характеристики» / «ключ: значение» в описании (b2).

Отдельного поля «атрибуты продавца» в данных нет: его роль играют такие блоки (CLAUDE.md, Контракты), поэтому
кандидаты парсера идут с source=seller_attrs. Блоки есть у 36–38% карточек без оффера и почти не встречаются
у офферов (EDA, раздел 2).

Ключ блока → attr_id по алиасам: имя атрибута и поля фида из реестра, алиасы из configs/normalize.yaml и ключи,
выученные по train — у офферов, где значение ключа совпало с эталоном атрибута.
"""
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from attrex.contracts import BuildContext, Candidate, Card, CategorySchema, Source, Step
from attrex.normalize.numbers import unit_key
from attrex.normalize.resources import NormalizeResources
from attrex.normalize.settings import NormalizeSettings
from attrex.normalize.text import fold, phrase_key
from attrex.normalize.values import ValueNormalizer
from attrex.registry import Registry

_LINE = re.compile(r"^[ \t]*[•\-–—*·]?[ \t]*([^:\n]{1,60}?)[ \t]*:[ \t]*([^\n]{1,200}?)[ \t.;]*$")
# несколько пар в одной строке: «Размер, мм: 200. Вес брутто, кг: 0,66.»
_INLINE_SPLIT = re.compile(r"(?<=[.;])[ \t]+(?=[A-ZА-ЯЁ][^:.\n]{1,40}:)")
_KEY_UNIT = re.compile(r"^(.*?)(?:,\s*|\s*\(\s*)([a-zа-я\"]{1,4})\.?\s*\)?$")   # «высота, см», «вес (кг)»
_MULTI_SPLIT = re.compile(r"\s*[,;]\s*")

_LEARN_SQL = """
    SELECT c.sku_id, c.description, g.attr_id, g.value
    FROM core_cards c JOIN core_splits s USING (sku_id) JOIN core_gold g USING (sku_id)
    WHERE s.fold IN (SELECT unnest(?)) AND g.value IS NOT NULL AND c.description IS NOT NULL
      AND regexp_matches(c.description, '(^|\\n)[^\\n:]{1,60}:\\s*\\S')
"""


@dataclass(frozen=True)
class KvPair:
    key: str              # свёрнутый ключ без единицы: «высота»
    unit: str | None      # единица из ключа: «высота, см» → «см»
    value: str            # значение как в тексте
    line: str             # исходная строка пары — evidence


def parse_pairs(text: str) -> list[KvPair]:
    """Пары «ключ: значение» из текста описания (по строкам; несколько пар в строке тоже разбираются)."""
    out = []
    for line in text.splitlines():
        for part in _INLINE_SPLIT.split(line):
            for m in _LINE.finditer(part):
                key, unit = _split_key_unit(m.group(1))
                if key and m.group(2).strip():
                    out.append(KvPair(key, unit, m.group(2).strip(), m.group(0).strip()))
    return out


def _split_key_unit(raw_key: str) -> tuple[str, str | None]:
    key = phrase_key(raw_key)
    m = _KEY_UNIT.match(fold(raw_key).strip())
    if m and unit_key(m.group(2)):
        return phrase_key(m.group(1)), m.group(2)
    return key, None


@dataclass(frozen=True)
class Alias:
    attr_id: str
    score: float          # точность алиаса: 1 у имени из реестра, доля совпадений у выученного


class KeyAliases:
    """Ключ блока → атрибут. Ручные алиасы важнее выученных."""

    def __init__(self, aliases: dict[str, Alias]):
        self._aliases = aliases

    def __getitem__(self, key: str) -> Alias | None:
        return self._aliases.get(key)

    def __len__(self) -> int:
        return len(self._aliases)

    @classmethod
    def manual(cls, registry: Registry, settings: NormalizeSettings, score: float) -> "KeyAliases":
        out: dict[str, Alias] = {}
        for attr_id, spec in registry.attrs.items():
            names = [spec.name_ru, *spec.feed_facet, *(n.removesuffix("_search") for n in spec.feed_search),
                     *settings.aliases.get(attr_id, ())]
            for name in names:
                key = _split_key_unit(re.sub(r"\(.*?\)", "", name))[0]
                if key:
                    out.setdefault(key, Alias(attr_id, score))
        return cls(out)

    def learn(self, rows: Iterable[tuple[str, dict[str, set[str]]]], normalizer: ValueNormalizer,
              min_support: int, min_precision: float) -> "KeyAliases":
        """rows: (описание, {attr_id: значения эталона}). Новый алиас — ключ, чьё значение в каноне совпадает
        с эталоном атрибута не реже min_precision из min_support и более карточек с эталоном этого атрибута.

        Сравниваем именно то значение, которое выдаст парсер: у «Размер (ШхГхВ): 60x50x85» в строке есть и высота,
        но парсер возьмёт первое число — такой ключ не должен стать алиасом высоты.
        """
        seen: Counter = Counter()
        hits: Counter = Counter()
        for description, gold in rows:
            for pair in parse_pairs(description):
                if pair.key in self._aliases:
                    continue
                for attr_id, values in gold.items():
                    seen[pair.key, attr_id] += 1
                    canon = normalizer.canonical(attr_id, pair.value, pair.unit)
                    if canon and any(normalizer.same(attr_id, canon.value, v) for v in values):
                        hits[pair.key, attr_id] += 1
        best: dict[str, Alias] = {}
        for (key, attr_id), n in seen.items():
            precision = hits[key, attr_id] / n
            if n >= min_support and precision >= min_precision and precision > getattr(best.get(key), "score", 0):
                best[key] = Alias(attr_id, round(precision, 3))
        return KeyAliases({**best, **self._aliases})


class KvParserExtractor:
    step = Step.PARSE

    def __init__(self, aliases: KeyAliases, res: NormalizeResources):
        self.aliases, self.res = aliases, res

    def extract(self, card: Card, schema: CategorySchema, prior: list[Candidate]) -> list[Candidate]:
        wanted = set(schema.required)
        pairs = parse_pairs(card.description or "") + [
            KvPair(*_split_key_unit(k), str(v), f"{k}: {v}") for k, v in card.seller_attrs.items()]
        return [c for pair in pairs for c in self._candidates(pair, wanted)]

    def _candidates(self, pair: KvPair, wanted: set[str]) -> Iterator[Candidate]:
        alias = self.aliases[pair.key]
        if alias is None or alias.attr_id not in wanted:
            return
        spec = self.res.registry[alias.attr_id]
        for raw in self._parts(alias.attr_id, pair.value, spec.multi):
            canon = self.res.normalizer.canonical(alias.attr_id, raw, pair.unit)
            if canon:
                yield Candidate(attr_id=alias.attr_id, value=canon.value, step=self.step, source=Source.SELLER_ATTRS,
                                score=round(alias.score * canon.score, 4), value_raw=raw, evidence=pair.line)

    def _parts(self, attr_id: str, value: str, multi: bool) -> list[str]:
        """Многозначный атрибут («Материал: полиэстер, хлопок») — по частям, если целиком значения нет в справочнике."""
        if not multi or self.res.dictionary.lookup(attr_id, value):
            return [value]
        return [p for p in _MULTI_SPLIT.split(value) if p]


def _learn_rows(ctx: BuildContext) -> list[tuple[str, dict[str, set[str]]]]:
    cards: dict[int, tuple[str, dict[str, set[str]]]] = {}
    for sku, description, attr_id, value in ctx.con.execute(_LEARN_SQL, [list(ctx.knowledge_folds)]).fetchall():
        cards.setdefault(sku, (description, defaultdict(set)))[1][attr_id].add(value)
    return list(cards.values())


def build(ctx: BuildContext) -> KvParserExtractor:
    res = ctx.resource(NormalizeResources.from_context)
    p = res.settings.parse
    aliases = KeyAliases.manual(res.registry, res.settings, p.alias_score)
    aliases = aliases.learn(_learn_rows(ctx), res.normalizer, p.learn_min_support, p.learn_min_precision)
    return KvParserExtractor(aliases, res)
