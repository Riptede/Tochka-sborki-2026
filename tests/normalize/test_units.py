import pytest

from attrex.normalize.units import canon_text, is_range, to_canonical_number


@pytest.mark.parametrize("value, unit, target, expected", [
    ("256", "ГБ", "GB", "256"),
    ("1", "ТБ", "GB", "1024"),
    ("512", "МБ", "GB", "0.5"),
    ("256 ГБ", None, "GB", "256"),
    ("8/256GB", None, "GB", None),          # «8» без единицы здесь не память: «/» — не единица
    ("6.9", '"', "in", "6.9"),
    ("6,9-дюймовый", None, "in", "6.9"),
    ('15.6"(39.6 см)', None, "in", "15.6"),
    ("127", "см", "in", "50"),              # диагональ ТВ в сантиметрах
    ("5000", "мАч", "mAh", "5000"),
    ("5000 мА*ч", None, "mAh", "5000"),
    ("5", "Ач", "mAh", "5000"),
    ("98.7", "г", "kg", "0.099"),
    ("90", "мм", "cm", "9"),
    ("3000-5000 мАч", None, "mAh", None),   # корзина фасета, а не значение
    ("6.6\" и более", None, "in", None),
    ("1000", "Вт", "kg", None),             # чужая единица
    (None, None, "GB", None),
])
def test_to_canonical_number(value, unit, target, expected):
    assert to_canonical_number(value, unit, target) == expected


def test_canon_text():
    assert canon_text("  Чёрный / Красный ") == "черный/красный"
    assert canon_text("Серый   космос") == "серый космос"
    assert canon_text("") is None


def test_is_range():
    assert is_range("от 9\" до 11\"")
    assert not is_range("6.7")
