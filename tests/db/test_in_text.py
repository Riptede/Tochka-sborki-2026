import pytest

from attrex.db.build import dedupe_tokens
from attrex.db.in_text import CardText, Locator

KINDS = {"color": ("category", None), "cpu": ("category", None), "weight": ("number", "kg"),
         "screen_diag": ("number", "in"), "model": ("text", None)}
SYN = {("color", "серый"): ["titanium grey", "grey"]}


def where(attr, value, title, description=""):
    return Locator(KINDS, SYN).locate(attr, value, CardText(title, description))


@pytest.mark.parametrize("attr, value, title, description, expected", [
    ("color", "черный", "Смартфон черного цвета", "", "title"),                  # с точностью до окончания
    ("color", "серый", "Смартфон Galaxy S25 Titanium Grey", "", "synonym"),
    ("color", "серый", "Смартфон", "Цвет: серый", "description"),
    ("cpu", "intel core i5 13420h 2.1 ггц", "Ноутбук Intel Core i5-13420H", "", "partial"),
    ("cpu", "apple m4", "Ноутбук Apple MacBook Air", "", "partial"),
    ("color", "белый", "Чехол", "Прозрачный силикон", "none"),
    ("weight", "0.099", "Наушники", "Вес: 99 г", "description"),                  # перевод граммов в кг
    ("weight", "2", "Чехол 2 в 1", "", "none"),                                    # однозначное число без единицы
    ("weight", "2", "Гантель 2 кг", "", "title"),
    ("screen_diag", "50", "Телевизор 127 см", "", "title"),                        # сантиметры диагонали
    ("model", "galaxy s25", "Смартфон Samsung Galaxy S25 8/256GB", "", "title"),
])
def test_locate(attr, value, title, description, expected):
    assert where(attr, value, title, description) == expected


@pytest.mark.parametrize("raw, expected", [
    ("apple m2 max apple m2 max 3.5 ггц", "apple m2 max 3.5 ггц"),
    ("intel core i3 i3-1115g4 3 ггц", "intel core i3-1115g4 3 ггц"),
    ("amd ryzen 5 ryzen 5 7535hs", "amd ryzen 5 7535hs"),
    ("intel core i5 12450h 2.0 ггц", "intel core i5 12450h 2.0 ггц"),
    (None, None),
])
def test_dedupe_tokens(raw, expected):
    assert dedupe_tokens(raw) == expected
