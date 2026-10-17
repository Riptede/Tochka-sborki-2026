import numpy as np
import pytest

from attrex.contracts import Candidate, Card, CategorySchema, Source, Step
from attrex.normalize.retrieval import CharNgramEncoder, RetrievalExtractor, RetrievalSettings, ValueIndex


def test_char_encoder_is_deterministic_and_normalized():
    enc = CharNgramEncoder(dim=512)
    a, b = enc.encode(["Серый", "серый"]), enc.encode(["серыи"])
    assert np.allclose(a[0], a[1]) and np.isclose(np.linalg.norm(a[0]), 1.0)
    assert float(a[0] @ b[0]) > 0.5                                   # опечатка — близко
    assert float(a[0] @ enc.encode(["латунь"])[0]) < 0.2


class DenseOnly:
    """Кодировщик без разреженного пути — как эмбеддинги API."""

    def __init__(self, inner):
        self.inner = inner

    def encode(self, texts):
        return self.inner.encode(texts)


def test_sparse_and_dense_paths_agree():
    values = ["черный", "белый", "серый"]
    sparse = ValueIndex(values, CharNgramEncoder(dim=512))
    dense = ValueIndex(values, DenseOnly(CharNgramEncoder(dim=512)))
    for (i, s), (j, d) in zip(sparse.best(["черныи", "бел"]), dense.best(["черныи", "бел"])):
        assert i == j and s == pytest.approx(d, abs=1e-5)


@pytest.fixture(scope="module")
def retrieval(res):
    enc = CharNgramEncoder(dim=1024)
    indexes = {"color": ValueIndex(["черный", "белый", "серый", "голубой"], enc)}
    return RetrievalExtractor(indexes, enc, RetrievalSettings(attrs=("color",), min_similarity=0.6), res)


def test_unknown_kv_value_maps_to_reference(retrieval):
    card = Card(1, "Рюкзак городской", "Цвет: черныи")
    prior = [Candidate("color", "черныи", Step.PARSE, Source.SELLER_ATTRS, 0.27, "черныи", "Цвет: черныи")]
    out = retrieval.extract(card, CategorySchema(None, ("color",)), prior)
    assert [(c.value, c.step, c.source, c.evidence) for c in out] == \
        [("черный", Step.RETRIEVAL, Source.SELLER_ATTRS, "Цвет: черныи")]


def test_closed_attr_is_skipped(retrieval):
    card = Card(1, "Рюкзак серыйй")
    prior = [Candidate("color", "серый", Step.RULES, Source.TITLE, 0.9, "серыйй", "серыйй")]
    assert retrieval.extract(card, CategorySchema(None, ("color",)), prior) == []
    assert [c.value for c in retrieval.extract(card, CategorySchema(None, ("color",)), [])] == ["серый"]
