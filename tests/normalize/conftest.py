"""Маленький справочник и выученные раскладки — без БД и без data/."""
import pytest

from attrex.normalize.dimensions import DimensionLayout
from attrex.normalize.resources import NormalizeResources
from attrex.normalize.title import TitleLayout

# (attr_id, написание, канон, голоса) — как строки справочника из эталона train
DICT_ROWS = [
    ("brand", "samsung", "samsung", 50), ("brand", "xiaomi", "xiaomi", 40), ("brand", "apple", "apple", 30),
    ("brand", "sleepteam", "sleepteam", 20), ("brand", "365home", "365home", 30), ("brand", "tp-link", "tp-link", 5),
    ("brand", "huawei", "huawei", 9),
    ("color", "черный", "черный", 100), ("color", "белый", "белый", 80), ("color", "голубой", "голубой", 20),
    ("color", "небесно-голубой", "голубой", 5), ("color", "titanium grey", "серый", 6), ("color", "серый", "серый", 30),
    ("color", "мятный", "мятный", 4), ("color", "мятный", "зеленый", 3),
    ("material", "полиэстер", "полиэстер", 30), ("material", "натуральная кожа", "натуральная кожа", 5),
    ("cpu", "intel core i5-13420h 2.1 ггц", "intel core i5-13420h 2.1 ггц", 12),
    ("cpu", "amd ryzen 5 7520u 2.8 ггц", "amd ryzen 5 7520u 2.8 ггц", 7),
    ("drive", "ssd 512 гб", "ssd 512 гб", 40), ("drive", "ssd 1 тб", "ssd 1 тб", 20),
    ("condition", "восстановленный", "восстановленный", 10), ("condition", "новый", "новый", 100),
]

TITLE_ROWS = [
    ("Смартфон Samsung Galaxy S25 8/256GB черный", "Samsung"),
    ("Смартфон Xiaomi Redmi 12 8/256GB небесно-голубой", "Xiaomi"),
    ("Смартфон Apple iPhone 15 128GB Blue", "Apple"),
    ("Ноутбук HUAWEI MateBook D16 Intel i9 16/1TB", "HUAWEI"),
    ("Ноутбук Xiaomi RedmiBook 15", "Xiaomi"),
    ("Ноутбук Apple MacBook Air 13", "Apple"),
    ("Матрас SleepTeam Orto Cocos 100х155", "SleepTeam"),
    ("Матрас SleepTeam Active 80х190", "SleepTeam"),
    ("Матрас SleepTeam Base 90х200", "SleepTeam"),
    ("Восстановленный смартфон Samsung Galaxy A56", "Samsung"),
    ("Восстановленный смартфон Apple iPhone 13", "Apple"),
    ("Восстановленный смартфон Xiaomi 13T", "Xiaomi"),
]

MATTRESS, PILLOW = 36480, 34350
DIM_ROWS = [
    (MATTRESS, "Матрас SleepTeam Orto 100х155", {"width": {"100"}, "height": {"19"}}),
    (MATTRESS, "Матрас SleepTeam Active 80х190", {"width": {"80"}}),
    (MATTRESS, "Матрас SleepTeam Base 90х200", {"width": {"90"}}),
    (PILLOW, "Подушка 30x30 см", {"width": {"30"}, "height": {"30"}}),
    (PILLOW, "Подушка 40x40 см", {"width": {"40"}, "height": {"40"}}),
    (PILLOW, "Подушка 45x45 см", {"width": {"45"}, "height": {"45"}}),
]


@pytest.fixture(scope="session")
def res() -> NormalizeResources:
    return NormalizeResources.from_rows(DICT_ROWS)


@pytest.fixture(scope="session")
def layout() -> TitleLayout:
    return TitleLayout.learn(TITLE_ROWS, min_count=2, compat_words=["для"])


@pytest.fixture(scope="session")
def dims() -> DimensionLayout:
    return DimensionLayout.learn(DIM_ROWS, ["width", "height", "depth"], min_support=2, min_rate=0.5)
