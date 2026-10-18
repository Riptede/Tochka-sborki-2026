"""Раскладка заголовка «<что за товар> <бренд> <модель>».

Так собран заголовок у большинства офферов фида: «Фотошторы 365home Бежевый фон…», «Смартфон Xiaomi Redmi 12 8/256GB».
Префиксы («смартфон», «подушка декоративная», «восстановленный ноутбук») выучены по train: это часть заголовка
до бренда оффера. Зная префикс, бренд ищем сразу после него, а модель — всё, что идёт после бренда.
"""
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from attrex.contracts import BuildContext
from attrex.normalize.dictionary import DictMatch, RefDictionary
from attrex.normalize.settings import load_settings
from attrex.normalize.text import TextField, Token, phrase_key

MAX_PREFIX_WORDS = 5

_ROWS_SQL = """
    SELECT c.title, o.vendor
    FROM raw_offers o JOIN core_cards c USING (sku_id) JOIN core_splits s USING (sku_id)
    WHERE s.fold IN (SELECT unnest(?)) AND o.vendor IS NOT NULL AND c.title IS NOT NULL
"""


@dataclass(frozen=True)
class TitleParts:
    brand: DictMatch | None     # бренд из справочника
    brand_start: int            # позиции бренда в заголовке; -1 — бренд не найден
    brand_end: int              # здесь начинается модель
    prefix_known: bool          # заголовок начинается с выученного префикса
    offset: int = 0             # сколько слов между префиксом и брендом

    @property
    def found(self) -> bool:
        return self.brand_start >= 0


class TitleLayout:
    def __init__(self, prefixes: Iterable[str], max_brand_offset: int = 2, compat_words: Iterable[str] = (),
                 brand_prefix_ratio: float = 0.0):
        self.prefixes = frozenset(prefixes)
        self.max_brand_offset = max_brand_offset
        self.compat_words = frozenset(compat_words)
        self.brand_prefix_ratio = brand_prefix_ratio

    @classmethod
    def learn(cls, rows: Iterable[tuple[str, str]], min_count: int = 3, **kwargs) -> "TitleLayout":
        """rows: (заголовок, vendor оффера). Префикс — слова заголовка до первого вхождения бренда."""
        counts: Counter = Counter()
        for title, vendor in rows:
            words, brand = phrase_key(title).split(), phrase_key(vendor).split()
            i = _find_sublist(words, brand)
            if 0 < i <= MAX_PREFIX_WORDS:
                counts[" ".join(words[:i])] += 1
        return cls([p for p, n in counts.items() if n >= min_count], **kwargs)

    @classmethod
    def from_context(cls, ctx: BuildContext) -> "TitleLayout":
        s = load_settings()
        rows = ctx.con.execute(_ROWS_SQL, [list(ctx.knowledge_folds)]).fetchall()
        return cls.learn(rows, s.min_prefix_count, max_brand_offset=s.max_brand_offset, compat_words=s.compat_words,
                         brand_prefix_ratio=s.brand_prefix_ratio)

    def prefix_len(self, tokens: list[Token]) -> int:
        """Сколько первых слов заголовка — известный префикс (самый длинный); хотя бы одно слово оставляем бренду."""
        for n in range(min(MAX_PREFIX_WORDS, len(tokens) - 1), 0, -1):
            if " ".join(t.text for t in tokens[:n]) in self.prefixes:
                return n
        return 0

    def parse(self, title: TextField, brands: RefDictionary) -> TitleParts:
        toks = title.tokens
        k = self.prefix_len(toks)
        for offset in range(self.max_brand_offset + 1):
            i = k + offset
            if i >= len(toks):
                break
            if i > 0 and toks[i - 1].text in self.compat_words:   # «чехол для Samsung» — совместимость, не бренд
                continue
            hit = self._brand_at(brands, toks, i)
            if hit:
                return TitleParts(hit, hit.start, hit.end, k > 0, offset)
        # бренда нет в справочнике, но префикс известен: бренд — слово сразу после префикса
        if 0 < k < len(toks) and toks[k].text not in self.compat_words:
            return TitleParts(None, toks[k].start, toks[k].end, True)
        return TitleParts(None, -1, -1, k > 0)

    def _brand_at(self, brands: RefDictionary, toks: list[Token], i: int) -> DictMatch | None:
        """Самое длинное написание бренда с токена i. Но если его начало — бренд во много раз частотнее
        («xiaomi» у 2 тыс. офферов, «xiaomi redmi» — у 10), бренд — начало, а остаток уже модель:
        в эталоне «xiaomi» + «redmi note 13 pro»."""
        hits = brands.matches_at("brand", toks, i)
        if not hits:
            return None
        if self.brand_prefix_ratio > 0:
            for h in hits[1:]:
                if h.support >= self.brand_prefix_ratio * hits[0].support:
                    return h
        return hits[0]


def _find_sublist(words: list[str], part: list[str]) -> int:
    if not part:
        return -1
    for i in range(len(words) - len(part) + 1):
        if words[i:i + len(part)] == part:
            return i
    return -1
