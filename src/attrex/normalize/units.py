"""Приведение значений к канону реестра: числа — в каноническую единицу, текст — к единому написанию.

Общая библиотека: ею пользуются b1 (эталон core_gold), b2 (правила, grounding) и b3 (проверка ответа LLM).
"""
import re

# единица в тексте (нижний регистр) → (каноническая единица реестра, множитель)
UNIT_TABLE: dict[str, tuple[str, float]] = {
    "гб": ("GB", 1), "gb": ("GB", 1), "г": ("kg", 0.001), "мб": ("GB", 1 / 1024), "mb": ("GB", 1 / 1024),
    "тб": ("GB", 1024), "tb": ("GB", 1024),
    '"': ("in", 1), "″": ("in", 1), "дюйм": ("in", 1), "дюйма": ("in", 1), "дюймов": ("in", 1), "inch": ("in", 1),
    "мач": ("mAh", 1), "ма*ч": ("mAh", 1), "mah": ("mAh", 1), "ач": ("mAh", 1000), "ah": ("mAh", 1000),
    "кг": ("kg", 1), "kg": ("kg", 1), "т": ("kg", 1000),
    "см": ("cm", 1), "cm": ("cm", 1), "мм": ("cm", 0.1), "mm": ("cm", 0.1), "м": ("cm", 100),
}
# единицы, которые для конкретной канонической единицы означают другую величину: «см» у диагонали — это не длина
CROSS = {("in", "см"): 1 / 2.54, ("in", "cm"): 1 / 2.54}

_NUM = re.compile(r"(\d+(?:[.,]\d+)?)\s*([a-zа-яё\"″*]+)?", re.IGNORECASE)
_RANGE = re.compile(r"\d\s*[-–—]\s*\d|\b(до|от|более|менее|свыше)\b", re.IGNORECASE)


def canon_text(s: str | None) -> str | None:
    """Единое написание текстового значения: регистр, ё→е, пробелы, разделитель «/»."""
    if s is None:
        return None
    s = str(s).strip().lower().replace("ё", "е")
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s)
    return s or None


# слова цвета, которые продавцы дописывают к модели; рус. — в форме канона (ё→е)
COLOR_WORDS = [
    "черный", "белый", "серый", "серебристый", "золотой", "золотистый", "синий", "голубой", "зеленый", "красный",
    "розовый", "фиолетовый", "желтый", "оранжевый", "бежевый", "коричневый", "графитовый", "бирюзовый", "мятный",
    "лавандовый", "темно-серый", "темно-синий", "светло-серый", "космический",
    "black", "white", "grey", "gray", "silver", "gold", "golden", "blue", "green", "red", "pink", "purple", "violet",
    "yellow", "orange", "beige", "brown", "graphite", "titanium", "space", "midnight", "starlight", "mint", "lavender",
    "jetblack", "onyx", "marble", "cobalt", "icyblue", "navy",
]
_COLOR_RE = re.compile(r"(?<![\w-])(" + "|".join(sorted(map(re.escape, COLOR_WORDS), key=len, reverse=True))
                       + r")(?![\w-])")
_BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]")
# начало «хвоста» с характеристиками: конфигурация памяти «12/256gb», «8/128», «1tb», или «/» в записи ноутбуков
_SPEC_TAIL = re.compile(r"\s\d+\s*/\s*\d+\s*(gb|tb|гб|тб)?\b.*$|\s\d+\s*(gb|tb|гб|тб)\b.*$|\s*/.*$|,.*$")


def canon_model(s: str | None) -> str | None:
    """Название модели без артикулов в скобках, конфигурации памяти, характеристик через «/» и цвета.

    «Galaxy S25 Ultra 12/256GB Titanium Gray» → «galaxy s25 ultra»,
    «TUF Gaming F15/15.6"/Core i7/16/1TB/черный» → «tuf gaming f15», «Zenbook 14 (90NB17R1-M006H0)» → «zenbook 14».
    """
    s = canon_text(s)
    if s is None:
        return None
    s = _BRACKETS.sub(" ", s)
    s = _SPEC_TAIL.sub("", s)
    s = _COLOR_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" -,")
    return s or None


def fmt_number(x: float) -> str:
    """Число строкой без лишних нулей: 256.0 → "256", 6.70 → "6.7"."""
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def is_range(value: str) -> bool:
    """«3000-5000 мАч», «до 4.9"», «6.6" и более» — корзина фасета, а не точное значение."""
    return bool(_RANGE.search(value))


def to_canonical_number(value: str | None, unit: str | None, target_unit: str) -> str | None:
    """Первое число значения в единице target_unit или None, если перевести нельзя.

    unit — атрибут @unit из фида (может быть None: тогда единица ищется в самом значении).
    Если единица не найдена, число считается уже записанным в target_unit.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or is_range(text):
        return None
    m = _NUM.search(text)
    if not m or text[m.end(1):m.end(1) + 1] == "/":   # «8/256GB» — составная запись, разбирают правила b2
        return None
    x = float(m.group(1).replace(",", "."))
    u = (unit or m.group(2) or "").strip().lower().replace("ё", "е")
    if not u:
        return fmt_number(x)
    if (target_unit, u) in CROSS:
        return fmt_number(x * CROSS[(target_unit, u)])
    canon, k = UNIT_TABLE.get(u, (None, None))
    if canon != target_unit:
        return None
    return fmt_number(x * k)
