from attrex.category.classifier import CategoryResolver, TitleCategoryClassifier, title_key
from attrex.contracts import Card, CategorySource

PHONES, CASES, LAPTOPS = 761, 900, 987
ROWS = [
    ("Смартфон Samsung Galaxy S25", PHONES), ("Смартфон Samsung Galaxy A56", PHONES),
    ("Смартфон Apple iPhone 15", PHONES), ("Смартфон Xiaomi Redmi 12", PHONES),
    ("Смартфон Xiaomi чехол-книжка", CASES),                     # шум: тот же ключ «смартфон xiaomi»
    ("Ноутбук HUAWEI MateBook D16", LAPTOPS), ("Ноутбук Apple MacBook Air", LAPTOPS),
    ("Чехол для смартфона Samsung", CASES),
]


def test_title_key_skips_condition():
    assert title_key("Восстановленный смартфон Apple iPhone 13", 2) == "смартфон apple"
    assert title_key("Ноутбук", 2) == "ноутбук"


def test_predict_two_words_then_one():
    clf = TitleCategoryClassifier.learn(ROWS)
    g = clf.predict("Смартфон Samsung Galaxy S24")
    assert (g.category_id, g.key, g.confidence) == (PHONES, "смартфон samsung", round(2 / 3, 4))
    g = clf.predict("Смартфон Honor 200")                        # двух слов нет в train — по одному
    assert (g.category_id, g.key) == (PHONES, "смартфон")
    assert clf.predict("Матрас SleepTeam") is None


def test_resolver_keeps_known_category():
    resolver = CategoryResolver(TitleCategoryClassifier.learn(ROWS))
    known = Card(1, "Смартфон Samsung", category_id=CASES, category_conf=1.0, category_source=CategorySource.FEED)
    assert resolver.resolve(known) is known
    guessed = resolver.resolve(Card(2, "Ноутбук Apple MacBook Pro"))
    assert guessed.category_id == LAPTOPS and guessed.category_source is CategorySource.TITLE
    assert 0 < guessed.category_conf < 1
