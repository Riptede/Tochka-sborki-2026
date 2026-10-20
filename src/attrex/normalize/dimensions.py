"""Размеры «AxB» и «AxBxC» в заголовке и в описании: какое число — ширина, какое — высота или глубина.

Порядок зависит от категории: у матраса «100х155» — ширина и длина, у подушки «30x30» — ширина и высота,
у фотоштор «290x260» — ширина и высота. Раскладка выучивается по train: для каждой позиции числа считаем,
с каким атрибутом эталона оно совпадает чаще всего. Для категорий без данных — общая раскладка по всем категориям.

В описании («Габаритные размеры 26,2х35,3х45,2 см», «Размеры: 2400×360×366 мм») размеры берём осторожнее: только
с единицей или после слов «размер», «габарит», и не размеры упаковки, экрана или фото — у описания своя раскладка.
Подпись порядка рядом с числами («(ШхВхГ)», «в×ш×г») важнее выученной раскладки.
"""
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
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
# описание тянем один раз на карточку, а не на каждую строку эталона: тексты длинные
_DESC_ROWS_SQL = """
    SELECT c.category_id, c.description, list(g.attr_id), list(g.value)
    FROM core_cards c JOIN core_splits s USING (sku_id) JOIN core_gold g USING (sku_id)
    WHERE s.fold IN (SELECT unnest(?)) AND g.attr_id IN (SELECT unnest(?)) AND g.value IS NOT NULL
      AND regexp_matches(c.description, '\\d\\s*[xхХX×*]\\s*\\d')
    GROUP BY c.sku_id, c.category_id, c.description
"""
_DESC_CUE = re.compile(r"(?:размер|габарит)[^.!?\n]{0,40}$")
# размеры не самого товара: упаковки, экрана, фото, отверстия
_DESC_SKIP = re.compile(r"(?:упаков|коробк|экран|диспле|разрешен|матриц|пиксел|фото|изображен|отверст|ниш[аеуи])"
                        r"[^.!?\n]{0,40}$")
_ORDER = re.compile(r"(?<![а-яa-z])([швг])\s*[xх×*]\s*([швг])(?:\s*[xх×*]\s*([швг]))?\W{0,12}(?:\w+\W{0,3}){0,2}$")
_ORDER_ATTR = {"ш": "width", "в": "height", "г": "depth"}


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


def find_description_dims(folded: str) -> list[DimsMention]:
    """Размеры товара в описании: с единицей или после «размер», «габарит»; без упаковки, экрана и фото."""
    out = []
    for d in find_dims(folded):
        before = folded[max(0, d.start - 50):d.start]
        if (d.unit or _DESC_CUE.search(before)) and not _DESC_SKIP.search(before):
            out.append(d)
    return out


def labeled_order(folded: str, d: DimsMention) -> tuple[str, ...] | None:
    """Подпись порядка перед размерами: «Размеры (ШхВхГ): 60x85x60» → (width, height, depth)."""
    m = _ORDER.search(folded[max(0, d.start - 40):d.start])
    if not m:
        return None
    letters = [g for g in m.groups() if g]
    if len(letters) != len(d.values) or len(set(letters)) != len(letters):
        return None
    return tuple(_ORDER_ATTR[x] for x in letters)


Layout = dict[int, tuple[str, float]]     # позиция числа → (attr_id, доля совпадений с эталоном)


class DimensionLayout:
    def __init__(self, layouts: dict[tuple[int | None, int], Layout]):
        self._layouts = layouts     # (category_id | None для общей, число размеров) → раскладка

    @classmethod
    def learn(cls, rows: Iterable[tuple[int | None, str, dict[str, set[str]]]], attrs: Sequence[str],
              min_support: int = 5, min_rate: float = 0.5,
              finder: Callable[[str], list[DimsMention]] = find_dims) -> "DimensionLayout":
        """rows: (category_id, заголовок или описание, {attr_id: значения эталона в см}); finder — какие размеры брать."""
        seen: Counter = Counter()
        hits: Counter = Counter()
        for cat, title, gold in rows:
            dims = finder(fold(title))
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

    @classmethod
    def description_from_context(cls, ctx: BuildContext) -> "DimensionLayout":
        """Раскладка размеров в описании: порядок там свой («габариты ВхШхГ»), отдельно от заголовка."""
        s, reg = load_settings(), load_registry()
        attrs = [a for a, spec in reg.attrs.items() if spec.unit == "cm"]
        rows = []
        for cat, desc, attr_ids, values in ctx.con.execute(_DESC_ROWS_SQL, [list(ctx.knowledge_folds), attrs]).fetchall():
            gold: dict[str, set[str]] = defaultdict(set)
            for a, v in zip(attr_ids, values):
                gold[a].add(v)
            rows.append((cat, desc, gold))
        return cls.learn(rows, attrs, s.dims_min_support, s.dims_min_rate, find_description_dims)

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
