import datetime
import pickle

import pyarrow.parquet as pq
import pytest

from attrex.db.feed import UnsafePickleError, feed_cached, parse_feed

from .conftest import FEED, OFFERS


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_parse_feed_modes_agree(tmp_path, fast, protocol):
    src = tmp_path / "skus.pkl"
    src.write_bytes(pickle.dumps(FEED, protocol=protocol))
    out = tmp_path / "interim"
    n_off, n_cat = parse_feed(src, out, fast=fast, log=lambda _: None)
    assert (n_off, n_cat) == (3, 3)
    assert feed_cached(out)
    offers = pq.read_table(out / "offers.parquet").to_pylist()
    assert [o["offer_id"] for o in offers] == [o["@id"] for o in OFFERS]
    assert offers[0]["category_id"] == "761"
    params = pq.read_table(out / "params.parquet").to_pylist()
    assert len(params) == 9 + 4 + 1          # одиночный param-словарь тоже разворачивается
    rom = next(p for p in params if p["param_name"] == "Встроенная память (ROM)_search")
    assert (rom["value"], rom["unit"]) == ("256", "ГБ")


@pytest.mark.parametrize("fast", [False, True])
def test_parse_feed_rejects_classes(tmp_path, fast):
    src = tmp_path / "skus.pkl"
    src.write_bytes(pickle.dumps({"yml_catalog": {"x": datetime.date(2026, 1, 1)}}))
    with pytest.raises((UnsafePickleError, pickle.UnpicklingError)):
        parse_feed(src, tmp_path / "out", fast=fast, log=lambda _: None)
