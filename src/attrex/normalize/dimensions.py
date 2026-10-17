"""Размеры «AxB» и «AxBxC» в заголовке: какое число — ширина, какое — высота или глубина.

Порядок зависит от категории: у матраса «100х155» — ширина и длина, у подушки «30x30» — ширина и высота,
у фотоштор «290x260» — ширина и высота. Раскладка выучивается по train: для каждой позиции числа считаем,
с каким атрибутом эталона оно совпадает чаще всего. Для категорий без данных — общая раскладка по всем категориям.
"""
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from attrex.contracts import BuildContext
from attrex.normalize.numbers import same_number
from attrex.normalize.settings import load_settings
from attrex.normalize.text import fold
from attrex.normalize.units import to_canonical_number
from attrex.registry import load_registry

_SEP = r"\s*[xх×*]\s*"
_NUM = r"(\d+(?:\.\d+)?)"
_DIMS = re.compile(rf"(?<![\w.]){_NUM}{_SEP}{_NUM}(?:{_SEP}{_NUM})?(?:\s*(мм|см|м|mm|cm)(?!\w))?")

_ROWS_SQL = """
    SELECT c.sku_id, c.category_id, c.title, g.attr_id, g.value
    FROM core_cards c JOIN core_splits s USING (sku_id) JOIN core_gold g USING (sku_id)
    WHERE s.fold IN (SELECT unnest(?)) AND g.attr_id IN (SELECT unnest(?)) AND g.value IS NOT NULL
      AND regexp_matches(c.title, '\\d\\s*[xхХX×*]\\s*\\d')
"""


@dataclass(frozen=True)
class DimsMention:
    values: tuple[str, ...]
    unit: str | None
    start: int
    end: int

    def value_cm(self, pos: int) -> str | None:
        """Число в сантиметрах; без единицы — уже сантиметры (так пишут почти все продавцы)."""
        return to_canonical_number(f"{self.values[pos]} {self.unit or 'см'}", None, "cm")


def find_dims(folded: str) -> list[DimsMention]:
    return [DimsMention(tuple(v for v in m.groups()[:3] if v), m.group(4), m.start(), m.end())
            for m in _DIMS.finditer(folded)]


Layout = dict[int, tuple[str, float]]     # позиция числа → (attr_id, доля совпадений с эталоном)


class DimensionLayout:
    def __init__(self, layouts: dict[tuple[int | None, int], Layout]):
        self._layouts = layouts     # (category_id | None для общей, число размеров) → раскладка

    @classmethod
    def learn(cls, rows: Iterable[tuple[int | None, str, dict[str, set[str]]]], attrs: Sequence[str],
              min_support: int = 5, min_rate: float = 0.5) -> "DimensionLayout":
        """rows: (category_id, заголовок, {attr_id: значения эталона в см})."""
        seen: Counter = Counter()
        hits: Counter = Counter()
        for cat, title, gold in rows:
            dims = find_dims(fold(title))
            if not dims:
                continue
            d = dims[0]
            for key in ((cat, len(d.values)), (None, len(d.values))):
                seen[key] += 1
                for pos in range(len(d.values)):
                    v = d.value_cm(pos)
                    for a in attrs:
                        if any(same_number(v, g) for g in gold.get(a, ())):
                            hits[key, pos, a] += 1
        by_key: dict = defaultdict(list)
        for (key, pos, a), h in hits.items():
            rate = h / seen[key]
            if seen[key] >= min_support and rate >= min_rate:
                by_key[key].append((rate, pos, a))
        return cls({key: _assign(cands, attrs) for key, cands in by_key.items()})

    @classmethod
    def from_context(cls, ctx: BuildContext) -> "DimensionLayout":
        s, reg = load_settings(), load_registry()
        attrs = [a for a, spec in reg.attrs.items() if spec.unit == "cm"]
        gold: dict[int, tuple[int | None, str, dict[str, set[str]]]] = {}
        for sku, cat, title, a, v in ctx.con.execute(_ROWS_SQL, [list(ctx.knowledge_folds), attrs]).fetchall():
            gold.setdefault(sku, (cat, title, defaultdict(set)))[2][a].add(v)
        return cls.learn(gold.values(), attrs, s.dims_min_support, s.dims_min_rate)

    def layout(self, category_id: int | None, n_values: int) -> Layout:
        return self._layouts.get((category_id, n_values)) or self._layouts.get((None, n_values), {})


def _assign(cands: list[tuple[float, int, str]], attrs: Sequence[str]) -> Layout:
    """Каждой позиции — один атрибут и каждому атрибуту — одна позиция; при равенстве — порядок реестра
    (у «30x30» первое число — ширина, второе — высота)."""
    order = {a: i for i, a in enumerate(attrs)}
    out: Layout = {}
    used: set[str] = set()
    for rate, pos, a in sorted(cands, key=lambda c: (-round(c[0], 2), c[1], order[c[2]])):
        if pos not in out and a not in used:
            out[pos] = (a, rate)
            used.add(a)
    return out
