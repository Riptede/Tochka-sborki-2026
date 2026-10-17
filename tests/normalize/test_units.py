import pytest

from attrex.normalize.units import canon_model, canon_text, is_range, to_canonical_number


@pytest.mark.parametrize("raw, expected", [
    ("Galaxy S25 Ultra 5G 12/256GB Черный", "galaxy s25 ultra 5g"),
    ("Galaxy S25 Ultra 12/256GB Titanium Gray", "galaxy s25 ultra"),
    ('TUF Gaming F15/15.6"/Core i7-11800H/16/1TB/RTX 3050 Ti/noOS/черный', "tuf gaming f15"),
    ("Zenbook 14 (90NB17R1-M006H0)", "zenbook 14"),
    ("MateBook D16 MCLG-X [53013YDL]", "matebook d16 mclg-x"),
    ("Redmi 13 green", "redmi 13"),
    ("iPhone 17 Pro Max", "iphone 17 pro max"),
    ("Galaxy S26+", "galaxy s26+"),
    ("Mi Band 4/Mi Band 3 розовый", "mi band 4"),
    ("Huawei MatePad 10.4 черный", "huawei matepad 10.4"),
    ("Galaxy S24 8/128GB Onyx Black, хороший", "galaxy s24"),
    (None, None),
])
def test_canon_model(raw, expected):
    assert canon_model(raw) == expected


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
