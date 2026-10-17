import pytest

from attrex.contracts import Card, CategorySchema, Source, Step
from attrex.normalize.rules import RulesExtractor, default_rules
from attrex.normalize.text import TextField

from .conftest import MATTRESS, PILLOW

PHONE = ("brand", "model", "storage", "screen_diag", "battery", "color")
LAPTOP = ("brand", "cpu", "ram", "drive", "screen_diag", "color")


@pytest.fixture(scope="module")
def rules(res, layout, dims):
    return RulesExtractor(default_rules(res, layout, dims))


def values(cands, source=None):
    out = {}
    for c in cands:
        if source is None or c.source is source:
            out.setdefault(c.attr_id, set()).add(c.value)
    return out


def test_title_layout(layout, res):
    parts = layout.parse(TextField(Source.TITLE, "Смартфон Samsung Galaxy S25"), res.dictionary)
    assert parts.prefix_known and parts.brand.value == "samsung"
    unknown = layout.parse(TextField(Source.TITLE, "Матрас Ormatek Flex 90х200"), res.dictionary)
    assert unknown.brand is None and unknown.found                      # бренда нет в справочнике
    compat = layout.parse(TextField(Source.TITLE, "Чехол для Samsung Galaxy"), res.dictionary)
    assert not compat.found                                             # «для Samsung» — не бренд


def test_dimension_layout_learned_per_category(dims):
    assert dims.layout(MATTRESS, 2) == {0: ("width", 1.0)}               # второе число матраса — длина
    assert {a for a, _ in dims.layout(PILLOW, 2).values()} == {"width", "height"}


def test_phone(rules):
    card = Card(sku_id=1, title="Смартфон Xiaomi Redmi 12 8/256GB небесно-голубой",
                description="Экран 6,79 дюйма. Аккумулятор ёмкостью 5000 мАч. Объём оперативной памяти — 8 Гб.",
                category_id=761)
    cands = rules.extract(card, CategorySchema(761, PHONE), [])
    got = values(cands)
    assert got["brand"] == {"xiaomi"}
    assert got["model"] == {"redmi 12"}
    assert got["storage"] == {"256"}
    assert got["color"] == {"голубой"}
    assert got["screen_diag"] == {"6.79"}
    assert got["battery"] == {"5000"}
    assert "ram" not in got                       # нет в схеме смартфона — не извлекаем
    assert all(c.step is Step.RULES and c.evidence for c in cands)


def test_laptop_memory_and_cpu(rules):
    card = Card(sku_id=2, title="Ноутбук HUAWEI MateBook D16 Intel Core i5-13420H 16/512GB",
                description='Экран 16". Оперативная память 16 ГБ, SSD 512 ГБ.', category_id=987)
    got = values(rules.extract(card, CategorySchema(987, LAPTOP), []))
    assert got["brand"] == {"huawei"}
    assert got["cpu"] == {"intel core i5-13420h 2.1 ггц"}
    assert got["ram"] == {"16"}                   # «512 ГБ» рядом с SSD — не оперативная память
    assert got["drive"] == {"ssd 512 гб"}
    assert got["screen_diag"] == {"16"}


def test_mattress_dimensions(rules):
    card = Card(sku_id=3, title="Матрас SleepTeam Orto Cocos 100х155, беспружинный, высота 19 см", category_id=MATTRESS)
    got = values(rules.extract(card, CategorySchema(MATTRESS, ("brand", "model", "width", "height", "weight")), []))
    assert got["width"] == {"100"}
    assert got["height"] == {"19"}                # из «высота 19 см», а не из «155»
    assert got["model"] == {"orto cocos 100х155"}
    assert "weight" not in got


def test_rules_candidates_pass_grounding(rules, res):
    card = Card(sku_id=4, title="Смартфон Samsung Galaxy S25 8/256GB черный",
                description="Батарея 4000 мА*ч, диагональ 6,2\"", category_id=761)
    cands = rules.extract(card, CategorySchema(761, PHONE), [])
    assert cands and all(res.grounding.check(card, c) for c in cands)


def test_year_is_not_weight(rules):
    card = Card(sku_id=5, title="Коврик автомобильный KIA Ceed 2012 г.", description="Вес 1,5 кг")
    got = values(rules.extract(card, CategorySchema(None, ("weight",)), []))
    assert got == {"weight": {"1.5"}}
