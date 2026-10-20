import pytest

from attrex.contracts import Card, CategorySchema, Source, Step
from attrex.normalize.dimensions import DimensionLayout, find_description_dims
from attrex.normalize.rules import RulesExtractor, default_rules
from attrex.normalize.text import TextField, fold

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


def test_description_dims_cues_and_skips():
    found = find_description_dims(fold("Прибор габаритами 342х342х95 мм. Экран 1920x1080. Упаковка 40x40x15 см."))
    assert [d.values for d in found] == [("342", "342", "95")]          # экран и упаковка — не размеры товара
    assert find_description_dims(fold("Набор 4 x 6 предметов")) == []    # без единицы и без «размер» — не размеры


def test_description_dims_labeled_order(res, layout, dims):
    rules = RulesExtractor(default_rules(res, layout, dims, DimensionLayout({})))
    card = Card(sku_id=10, title="Стиральная машина Hi WM610W", category_id=1,
                description="Размеры (ВхШхГ): 85х59,5х45 см. Загрузка 6 кг.")
    got = values(rules.extract(card, CategorySchema(1, ("width", "height", "depth")), []), Source.DESCRIPTION)
    assert got == {"height": {"85"}, "width": {"59.5"}, "depth": {"45"}}


def test_description_dims_layout_learned():
    rows = [(5, f"Габаритные размеры {h}x{w}x{d} см.", {"height": {h}, "width": {w}, "depth": {d}})
            for h, w, d in (("26.2", "45.2", "35.3"), ("10", "30", "20"), ("12", "40", "25"))]
    learned = DimensionLayout.learn(rows, ["width", "height", "depth"], min_support=2, finder=find_description_dims)
    assert learned.layout(5, 3) == {0: ("height", 1.0), 1: ("width", 1.0), 2: ("depth", 1.0)}


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


def test_keyword_before_number_wins_ties(rules):
    card = Card(sku_id=6, title="Фотошторы 365home", description="Высота 270 см, ширина одного полотна - 145 см.")
    got = values(rules.extract(card, CategorySchema(None, ("width", "height")), []))
    assert got == {"height": {"270"}, "width": {"145"}}


def test_rules_do_not_depend_on_hash_seed():
    """Порядок обхода множеств строк зависит от PYTHONHASHSEED: правила не должны от него зависеть."""
    import subprocess
    import sys
    code = ("from tests.normalize.conftest import DICT_ROWS; from attrex.normalize.resources import NormalizeResources;"
            "from attrex.normalize.rules import UnitNumberRule; from attrex.normalize.text import CardText;"
            "from attrex.contracts import Card;"
            "r = UnitNumberRule(NormalizeResources.from_rows(DICT_ROWS));"
            "c = Card(1, 'x', 'Высота 270 см, ширина 145 см, глубина 40 см, вес 2 кг');"
            "print(sorted((x.attr_id, x.value) for x in r.apply(CardText.of(c), frozenset({'width','height','depth','weight'}))))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**__import__("os").environ, "PYTHONHASHSEED": str(seed)}).stdout for seed in (1, 2, 3)}
    assert len(outs) == 1


def test_year_is_not_weight(rules):
    card = Card(sku_id=5, title="Коврик автомобильный KIA Ceed 2012 г.", description="Вес 1,5 кг")
    got = values(rules.extract(card, CategorySchema(None, ("weight",)), []))
    assert got == {"weight": {"1.5"}}


@pytest.mark.parametrize("title, description, expected", [
    ("Домкрат AutoProfi гидравлический 3т (DP-30HL)", "", {}),                  # грузоподъёмность
    ("Плита электрическая Мечта 341Т белая", "", {}),                            # модель
    ("Домкрат подкатной", "Грузоподъемность 2 т, вес 15 кг", {"weight": {"15"}}),  # «вес» после тонн — не про них
    ("Прицеп легковой", "Полная масса 0,75 т", {"weight": {"750"}}),             # ключевое слово перед числом
])
def test_tons_need_weight_keyword_before(rules, title, description, expected):
    card = Card(sku_id=7, title=title, description=description)
    assert values(rules.extract(card, CategorySchema(None, ("weight",)), [])) == expected


def test_color_in_title_tail_scores_higher(rules):
    schema = CategorySchema(None, ("color",))
    tail = rules.extract(Card(8, "Подушка декоративная 365home 40x40 голубой"), schema, [])
    inner = rules.extract(Card(9, "Подушка декоративная 365home голубой орнамент 40x40"), schema, [])
    assert [c.value for c in tail] == [c.value for c in inner] == ["голубой"]
    assert inner[0].score < tail[0].score                     # в середине заголовка цвет — часто узор или название


def test_color_word_inside_brand_is_not_color(dims):
    from attrex.normalize.resources import NormalizeResources
    from attrex.normalize.title import TitleLayout

    res = NormalizeResources.from_rows([("brand", "red line", "red line", 30), ("color", "голубой", "голубой", 30)])
    layout = TitleLayout.learn([("Чехол Red Line для iPhone", "Red Line")] * 3, min_count=2)
    rules = RulesExtractor(default_rules(res, layout, dims))
    got = values(rules.extract(Card(10, "Чехол Red Line силикон голубой"), CategorySchema(None, ("color",)), []))
    assert got == {"color": {"голубой"}}                      # «red» из бренда — не «красный»


def test_brand_prefix_far_more_frequent_wins(dims):
    from attrex.normalize.resources import NormalizeResources
    from attrex.normalize.title import TitleLayout

    res = NormalizeResources.from_rows([("brand", "xiaomi", "xiaomi", 2000), ("brand", "xiaomi redmi", "xiaomi redmi", 10),
                                        ("brand", "tp-link", "tp-link", 50)])
    rows = [("Смартфон Xiaomi Redmi 12", "Xiaomi")] * 3
    title = TextField(Source.TITLE, "Смартфон Xiaomi Redmi Note 13 Pro 8/256GB")
    assert TitleLayout.learn(rows, min_count=2).parse(title, res.dictionary).brand.value == "xiaomi redmi"
    layout = TitleLayout.learn(rows, min_count=2, brand_prefix_ratio=10)
    assert layout.parse(title, res.dictionary).brand.value == "xiaomi"
    got = values(RulesExtractor(default_rules(res, layout, dims)).extract(
        Card(11, title.text), CategorySchema(761, ("brand", "model")), []))
    assert got == {"brand": {"xiaomi"}, "model": {"redmi note 13 pro"}}


def test_model_after_brand_with_dot_and_symbol_brand(dims):
    from attrex.normalize.resources import NormalizeResources
    from attrex.normalize.title import TitleLayout

    res = NormalizeResources.from_rows([("brand", "leather co.", "leather co.", 30)])
    rows = [("Кардхолдер Leather Co. для iPhone 15", "Leather Co.")] * 3 + [("Удлинитель электрический Pilot 3", "Pilot")] * 3
    rules = RulesExtractor(default_rules(res, TitleLayout.learn(rows, min_count=2), dims))
    schema = CategorySchema(None, ("brand", "model"))
    got = values(rules.extract(Card(13, "Кардхолдер Leather Co. для iPhone 14 Pro Max белый"), schema, []))
    assert got == {"brand": {"leather co."}, "model": {"для iphone 14 pro max"}}    # точка бренда — не модель
    title = "Удлинитель электрический ### С плоской вилкой 3 розетки с кнопкой, 45.0 метров."
    got = values(rules.extract(Card(14, title), schema, []))
    assert got == {"brand": {"###"}, "model": {"с плоской вилкой 3 розетки с кнопкой"}}   # бренд — символы, не «С»


def test_assumed_ssd_cannot_pass_grounding(rules, res):
    """«16/512GB» не называет тип накопителя: SSD достроен правилом — значение только на ручную проверку."""
    card = Card(sku_id=12, title="Ноутбук HUAWEI MateBook D16 16/512GB", category_id=987)
    drive = [c for c in rules.extract(card, CategorySchema(987, LAPTOP), []) if c.attr_id == "drive"]
    assert [c.value for c in drive] == ["ssd 512 гб"]
    assert not res.grounding.check(card, drive[0])
