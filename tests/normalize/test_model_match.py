from attrex.contracts import Candidate, Card, CategorySchema, Source, Step
from attrex.normalize.grounding import Grounding
from attrex.normalize.model_match import ModelKnowledge, ModelMatchExtractor, ModelMatchSettings

KB = ModelKnowledge.from_rows([
    ("huawei", "pura 80", "screen_diag", "6.6", 5), ("huawei", "pura 80", "battery", "5170", 4),
    ("huawei", "pura 80", "battery", "5000", 1), ("huawei", "pura 80", "color", "черный", 2),
    ("huawei", "pura 80", "color", "белый", 2),                            # мнения разошлись — не берём
])
CARD = Card(1, "Восстановленный смартфон HUAWEI Pura 80 12/256GB Gold, отличный", "Внешний вид: «Отличный».")
PRIOR = [Candidate("brand", "huawei", Step.RULES, Source.TITLE, 0.9, "HUAWEI", "HUAWEI"),
         Candidate("model", "pura 80", Step.RULES, Source.TITLE, 0.7, "Pura 80 12/256GB Gold", "Pura 80 12/256GB Gold")]
SCHEMA = CategorySchema(761, ("brand", "model", "screen_diag", "battery", "color"))


def test_values_from_same_model():
    ex = ModelMatchExtractor(KB, ModelMatchSettings())
    out = {c.attr_id: c for c in ex.extract(CARD, SCHEMA, PRIOR)}
    assert set(out) == {"screen_diag", "battery"}
    assert out["battery"].value == "5170" and out["battery"].score == round(0.6 * 0.8, 4)
    assert all(c.step is Step.MODEL_MATCH and c.source is Source.OTHER_CARD for c in out.values())
    assert out["battery"].evidence == "Pura 80 12/256GB Gold"           # фрагмент этой карточки


def test_skips_found_attrs_and_unknown_models(res):
    ex = ModelMatchExtractor(KB, ModelMatchSettings())
    found = PRIOR + [Candidate("battery", "5100", Step.RULES, Source.DESCRIPTION, 0.8, "5100 мАч", "5100 мАч")]
    assert "battery" not in {c.attr_id for c in ex.extract(CARD, SCHEMA, found)}
    assert ex.extract(CARD, SCHEMA, PRIOR[:1]) == []                      # модель не опознана


def test_model_match_never_passes_grounding(res):
    cand = ModelMatchExtractor(KB, ModelMatchSettings()).extract(CARD, SCHEMA, PRIOR)[0]
    assert not Grounding(res.normalizer).check(CARD, cand)                 # значения нет в тексте этой карточки
