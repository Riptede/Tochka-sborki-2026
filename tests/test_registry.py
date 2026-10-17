from attrex.contracts import PREDICTION_COLUMNS, Prediction, Source, Status, Step
from attrex.registry import load_registry


def test_registry_loads_and_required_known():
    reg = load_registry()
    assert reg.required[761] == ("brand", "model", "storage", "screen_diag", "battery", "color")
    assert reg.required[987] == ("brand", "cpu", "ram", "drive", "screen_diag", "color")
    for attrs in reg.required.values():
        assert set(attrs) <= set(reg.attrs)


def test_numbers_have_units():
    reg = load_registry()
    for spec in reg.attrs.values():
        if spec.kind == "number":
            assert spec.unit, spec.attr_id


def test_feed_names_map_to_single_attr():
    reg = load_registry()
    seen = {}
    for a, s in reg.attrs.items():
        for name in (*s.feed_facet, *s.feed_search):
            assert name not in seen, f"{name} у {seen.get(name)} и {a}"
            seen[name] = a
    assert reg.by_feed_name()["Встроенная память (ROM)_search"] == "storage"


def test_placeholders():
    reg = load_registry()
    assert reg.is_placeholder("Другие цвета ")
    assert reg.is_placeholder(None)
    assert not reg.is_placeholder("черный")


def test_prediction_row_matches_columns():
    p = Prediction(sku_id=1, category_id=761, attr_id="color", value="черный", unit=None, confidence=0.9,
                   status=Status.ACCEPTED, step=Step.RETRIEVAL, source=Source.TITLE)
    row = p.to_row()
    assert set(row) == set(PREDICTION_COLUMNS)
    assert row["step"] == "retrieval"


def test_extractor_protocol_shape():
    from attrex.contracts import Candidate, Card, CategorySchema, Extractor

    class Dummy:
        step = Step.RULES

        def extract(self, card: Card, schema: CategorySchema, prior: list[Candidate]) -> list[Candidate]:
            return [Candidate(attr_id="brand", value="Samsung", step=self.step, source=Source.TITLE, score=1.0,
                              evidence="Samsung")]

    ex: Extractor = Dummy()
    out = ex.extract(Card(sku_id=1, title="Смартфон Samsung"), CategorySchema(761, ("brand",)), [])
    assert out[0].evidence in "Смартфон Samsung"
