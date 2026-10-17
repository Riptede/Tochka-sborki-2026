"""Ключи сравнения значений, у которых канон длиннее, чем значение пишут в тексте.

Эталон процессора — «intel core i5-13420h 2.1 ггц», а в тексте — «i5-13420H»; эталон накопителя — «ssd 512 гб»,
а в тексте — «512GB SSD». Справочник сравнивает такие значения по ключу: «i5 13420h», «ssd 512».
Какой ключ у какого атрибута — configs/normalize.yaml, раздел key_functions.
"""
import re
from collections.abc import Callable

from attrex.normalize.numbers import find_numbers
from attrex.normalize.text import fold, tokenize

KeyFn = Callable[[str], str | None]

_FREQ = re.compile(r"\d+(?:\.\d+)?\s*(?:ггц|ghz|мгц|mhz)")
_CORES = re.compile(r"(?<!\w)\d+\s*x(?!\w)")          # «2 x 4.47ггц» — число ядер у частоты
# производители и служебные слова: в тексте их часто опускают («Core i5-13420H», «Ryzen 5 7520U»)
_CPU_NOISE = {"процессор", "intel", "amd", "apple", "qualcomm", "mediatek", "samsung", "core", "ryzen", "®", "™"}
_DRIVE_TYPES = {"ssd": "ssd", "hdd": "hdd", "emmc": "emmc", "nvme": "ssd", "твердотельный": "ssd", "жесткий": "hdd"}


def cpu_key(s: str) -> str | None:
    """Модель процессора без производителя и частоты: «Intel Core i5-13420H 2.1 ГГц» → «i5 13420h».

    Ключ без цифр («huawei») или из одних цифр («3») — не модель процессора, а случайное слово текста.
    """
    s = _CORES.sub(" ", _FREQ.sub(" ", fold(s)))
    words = []
    for t in tokenize(s):
        words += [w for w in t.text.split("-") if w and w not in _CPU_NOISE]
    key = " ".join(words)
    if not any(c.isdigit() for c in key) or key.replace(" ", "").replace(".", "").isdigit():
        return None
    return key


def drive_key(s: str) -> str | None:
    """Тип и объём накопителя в ГБ: «SSD 512 ГБ», «512GB SSD» → «ssd 512». Два объёма во фрагменте — не ключ."""
    folded = fold(s)
    kinds = {_DRIVE_TYPES[t.text] for t in tokenize(folded) if t.text in _DRIVE_TYPES}
    sizes = [m for m in find_numbers(folded) if m.unit]
    if len(kinds) != 1 or len(sizes) != 1:
        return None
    size = sizes[0].to("GB")
    return f"{kinds.pop()} {size}" if size else None


KEY_FUNCTIONS: dict[str, KeyFn] = {"cpu": cpu_key, "drive": drive_key}
