from attrex.contracts import Candidate, Card, Source, Step
from attrex.normalize.text import fold, tokenize


def test_dictionary_lookup_synonyms_and_votes(res):
    d = res.dictionary
    assert d.lookup("color", "Titanium Grey").value == "серый"          # написание продавца → канон
    assert d.lookup("color", "Black").value == "черный"                 # ручной синоним
    assert d.lookup("color", "черного").value == "черный"               # по основе слова
    m = d.lookup("color", "Мятный")
    assert (m.value, round(m.score, 2), m.support) == ("мятный", 0.57, 7)   # 4 голоса из 7
    assert d.lookup("brand", "черный") is None
    assert d.lookup("cpu", "Core i5-13420H").value == "intel core i5-13420h 2.1 ггц"   # по ключу модели


def test_tie_does_not_depend_on_row_order():
    from attrex.normalize.dictionary import RefDictionary

    rows = [("color", "графит", "серый", 3), ("color", "графит", "черный", 3)]
    a = RefDictionary.from_rows(rows).lookup("color", "графит").value
    b = RefDictionary.from_rows(rows[::-1]).lookup("color", "графит").value
    assert a == b == "серый"


def test_dictionary_find_longest_match(res):
    text = fold("Цвет: небесно-голубой, вставки Titanium Grey")
    found = [(m.value, text[m.start:m.end]) for m in res.dictionary.find("color", tokenize(text))]
    assert found == [("голубой", "небесно-голубой"), ("серый", "titanium grey")]


def test_canonical_values(res):
    n = res.normalizer
    assert n.canonical("storage", "1", "ТБ").value == "1024"
    assert n.canonical("color", "Titanium Grey").value == "серый"
    unknown = n.canonical("color", "лавандовый туман")
    assert (unknown.value, unknown.known) == ("лавандовый туман", False)
    assert n.canonical("color", "другие цвета") is None                # заглушка — пропуск
    assert n.canonical("model", "Galaxy S25 Ultra 12/256GB Titanium Gray").value == "galaxy s25 ultra"


def test_same_value(res):
    n = res.normalizer
    assert n.same("screen_diag", "15.59", "15.6")
    assert n.same("cpu", "intel core i3-1215u 1.2 ггц", "intel core i3-1215u 0.9 ггц")   # частота не в счёт
    assert not n.same("cpu", "intel core i3-1215u 1.2 ггц", "intel core i5-13420h 2.1 ггц")
    assert not n.same("color", "черный", "белый")
    assert not n.same("color", None, "белый")


def test_supports_after_normalization(res):
    n = res.normalizer
    assert n.supports("storage", "8/256GB", "256")
    assert n.supports("screen_diag", '15.6"(39.6 см)', "15.6")
    assert n.supports("color", "Titanium Grey", "серый")
    assert n.supports("cpu", "Intel Core i5-13420H", "intel core i5-13420h 2.1 ггц")
    assert n.supports("model", "Galaxy S25 8/256GB черный", "galaxy s25")
    assert not n.supports("storage", "8/256GB", "512")
    assert not n.supports("color", "черный", "белый")


def test_grounding_requires_evidence_from_this_card(res):
    card = Card(sku_id=1, title="Смартфон Samsung Galaxy S25 8/256GB", description="Цвет корпуса — Titanium  Grey.")
    ok = Candidate("color", "серый", Step.RULES, Source.DESCRIPTION, 0.6, evidence="titanium grey")
    foreign = Candidate("color", "серый", Step.LLM, Source.DESCRIPTION, 0.9, evidence="серый")
    no_evidence = Candidate("storage", "256", Step.LLM, Source.TITLE, 0.9)
    wrong_value = Candidate("storage", "512", Step.RULES, Source.TITLE, 0.9, evidence="8/256GB")
    g = res.grounding
    assert g.check(card, ok)
    assert not g.check(card, foreign)        # «серый» в тексте карточки нет
    assert not g.check(card, no_evidence)
    assert not g.check(card, wrong_value)
