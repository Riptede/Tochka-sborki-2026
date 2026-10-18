import pytest

from attrex.contracts import Card, CategorySchema, Source, Step
from attrex.normalize.kv_parser import KeyAliases, KvParserExtractor, parse_pairs

DESCRIPTION = """Рюкзак UNIMALL с канвасом.

Характеристики:
• Цвет: Titanium Grey
• Материал: полиэстер, натуральная кожа
• Высота, см: 45
• Вес (кг): 0,8
• Внешний вид: «Отличный»
Размер, мм: 200. Ширина изделия, см: 30."""


def test_parse_pairs_lines_units_and_inline():
    pairs = {p.key: (p.unit, p.value) for p in parse_pairs(DESCRIPTION)}
    assert pairs["цвет"] == (None, "Titanium Grey")
    assert pairs["высота"] == ("см", "45")
    assert pairs["вес"] == ("кг", "0,8")
    assert pairs["размер"] == ("мм", "200")              # две пары в одной строке
    assert pairs["ширина изделия"] == ("см", "30")
    assert "характеристики" not in pairs                  # заголовок блока без значения


@pytest.fixture(scope="module")
def parser(res):
    return KvParserExtractor(KeyAliases.manual(res.registry, res.settings, 0.9), res)


def test_extract_known_keys_into_canon(parser):
    card = Card(sku_id=1, title="Рюкзак UNIMALL -", description=DESCRIPTION)
    schema = CategorySchema(None, ("color", "material", "height", "weight", "width", "brand"))
    got = {}
    for c in parser.extract(card, schema, []):
        assert c.step is Step.PARSE and c.source is Source.SELLER_ATTRS and c.evidence in DESCRIPTION
        got.setdefault(c.attr_id, set()).add(c.value)
    assert got == {"color": {"серый"}, "material": {"полиэстер", "натуральная кожа"}, "height": {"45"},
                   "weight": {"0.8"}, "width": {"30"}}


def test_candidates_pass_grounding(parser, res):
    card = Card(sku_id=1, title="Рюкзак", description=DESCRIPTION)
    cands = parser.extract(card, CategorySchema(None, ("color", "height", "weight")), [])
    assert cands and all(res.grounding.check(card, c) for c in cands)


def test_learned_alias(res):
    manual = KeyAliases.manual(res.registry, res.settings, 0.9)
    assert manual["высота матраса"] is None
    rows = [(f"Высота матраса: {h} см\nЖесткость: средняя", {"height": {str(h)}}) for h in (16, 18, 20, 22)]
    rows.append(("Высота матраса: 25 см", {"height": {"24"}}))
    learned = manual.learn(rows, res.normalizer, min_support=4, min_precision=0.7)
    assert learned["высота матраса"].attr_id == "height"
    assert learned["высота матраса"].score == 0.8          # 4 из 5 совпали с эталоном
    assert learned["жесткость"] is None                   # не подтверждает ни один атрибут
    assert learned["цвет"].attr_id == "color"             # ручные алиасы на месте
