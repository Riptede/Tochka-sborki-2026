"""Числа с единицами в свёрнутом тексте: «256gb», «5000 ма*ч», «6.7"», «17.3-дюймовый», «290x260 см».

Перевод в каноническую единицу — общий с эталоном: `units.to_canonical_number`.
"""
import re
from dataclasses import dataclass

from attrex.normalize.units import CROSS, UNIT_TABLE, to_canonical_number

# число не продолжает слово или другое число («s25», «v1.2»), но может идти после знака размера («290x260»)
_MENTION = re.compile(r"(?:(?<![\w.,])|(?<=[xх×*]))(\d+(?:\.\d+)?)(?:\s*-?\s*([a-zа-я\"″'*]+))?")
_RANGE_BEFORE = re.compile(r"\d\s*[-–—]\s*$")
# написания единиц, которых нет в UNIT_TABLE: формы слова «дюйм» и кавычки
_UNIT_ALIASES = {"''": '"', "inches": '"', "дюймовый": '"', "дюймовая": '"', "дюймовым": '"', "дюймовом": '"'}


@dataclass(frozen=True)
class NumberMention:
    number: str            # как в тексте, с точкой
    unit: str | None       # ключ UNIT_TABLE или None, если единицы нет или она не наша
    start: int
    end: int

    def to(self, target_unit: str) -> str | None:
        """Значение в канонической единице атрибута; без единицы число считается уже в ней."""
        return to_canonical_number(f"{self.number} {self.unit}" if self.unit else self.number, None, target_unit)


def unit_key(word: str | None) -> str | None:
    if not word:
        return None
    word = _UNIT_ALIASES.get(word, word)
    if word.startswith("дюйм"):
        return '"'
    return word if word in UNIT_TABLE else None


def find_numbers(folded: str) -> list[NumberMention]:
    """Все числа текста. В диапазоне «3000-5000 мАч» верхняя граница пропускается, а нижняя остаётся без единицы:
    корзина фасета не должна стать значением."""
    out = []
    for m in _MENTION.finditer(folded):
        if _RANGE_BEFORE.search(folded, 0, m.start()):
            continue
        unit = unit_key(m.group(2))
        end = m.end() if unit else m.end(1)
        out.append(NumberMention(m.group(1), unit, m.start(1), end))
    return out


def unit_fits(unit: str | None, target_unit: str) -> bool:
    """Единица из текста переводится в каноническую единицу атрибута (в том числе «см» → дюймы у диагонали)."""
    if unit is None:
        return False
    return UNIT_TABLE[unit][0] == target_unit or (target_unit, unit) in CROSS


def is_cross_unit(unit: str, target_unit: str) -> bool:
    """Единица другой величины, которую для атрибута можно перевести: «см» у диагонали."""
    return (target_unit, unit) in CROSS


def same_number(a: str | None, b: str | None, rel: float = 0.01) -> bool:
    """Числа в одной единице равны с точностью округления (15.59 ≈ 15.6 после перевода из сантиметров)."""
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(x - y) <= max(rel * max(abs(x), abs(y)), 0.051)


def numbers_in(folded: str, target_unit: str) -> list[str]:
    """Канонические значения всех чисел текста, которые можно перевести в target_unit."""
    out = []
    for m in find_numbers(folded):
        if m.unit is None or unit_fits(m.unit, target_unit):
            v = m.to(target_unit)
            if v is not None:
                out.append(v)
    return out
