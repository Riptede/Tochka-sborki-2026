import pytest

from attrex.normalize.keys import cpu_key, drive_key
from attrex.normalize.numbers import find_numbers, numbers_in, same_number
from attrex.normalize.text import fold, phrase_key, stem, tokenize


def test_fold_keeps_length_and_positions():
    s = "Экран 6,7\" Чёрный\xa0Titan"
    f = fold(s)
    assert len(f) == len(s)
    assert f == 'экран 6.7" черный titan'
    i = f.index("черный")
    assert s[i:i + 6] == "Чёрный"
    assert fold("6,67‑дюймовым") == "6.67-дюймовым"          # неразрывный дефис — как обычный


def test_tokens_and_phrase_key():
    assert [t.text for t in tokenize(fold("TP-Link Galaxy S26+ i5-13420H 6.7"))] == \
        ["tp-link", "galaxy", "s26+", "i5-13420h", "6.7"]
    assert phrase_key("Серый / Космос") == "серый космос"


@pytest.mark.parametrize("word, expected", [
    ("черного", "черн"), ("черный", "черн"), ("полиэстера", "полиэстер"), ("стали", "стал"), ("сталь", "стал"),
    ("s25", "s25"), ("кот", "кот"),
])
def test_stem(word, expected):
    assert stem(word) == expected


def test_find_numbers_units_and_ranges():
    found = [(m.number, m.unit) for m in find_numbers(fold('8/256GB, экран 6,7", 5000 мА*ч, 17,3-дюймовый'))]
    assert found == [("8", None), ("256", "gb"), ("6.7", '"'), ("5000", "ма*ч"), ("17.3", '"')]
    assert [(m.number, m.unit) for m in find_numbers("3000-5000 мач")] == [("3000", None)]   # корзина, не значение
    assert numbers_in(fold("290x260 см"), "cm") == ["290", "260"]
    assert numbers_in("s25 ultra", "GB") == []


def test_same_number():
    assert same_number("15.59", "15.6")
    assert same_number("256", "256.0")
    assert not same_number("128", "256")
    assert not same_number(None, "1")


@pytest.mark.parametrize("s, expected", [
    ("Intel Core i5-13420H 2.1 ГГц", "i5 13420h"),
    ("i5-13420H", "i5 13420h"),
    ("AMD Ryzen 5 7520U", "5 7520u"),
    ("qualcomm snapdragon 8 elite 2 x 4.47ггц + 6 x 3.53ггц", "snapdragon 8 elite"),
])
def test_cpu_key(s, expected):
    assert cpu_key(s) == expected


@pytest.mark.parametrize("s, expected", [
    ("SSD 512 ГБ", "ssd 512"), ("512GB SSD", "ssd 512"), ("ssd 1 тб", "ssd 1024"),
    ("16GB RAM SSD 512GB", None),   # два объёма — непонятно, какой из них накопитель
    ("512 ГБ", None),
])
def test_drive_key(s, expected):
    assert drive_key(s) == expected
