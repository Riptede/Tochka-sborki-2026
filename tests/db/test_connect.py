import duckdb
import pytest

from attrex.db import DbError, connect, require_raw


def test_missing_db_hints_build_and_demo(tmp_path):
    with pytest.raises(DbError) as e:
        connect(tmp_path / "attrex.duckdb")
    assert e.value.code.startswith("ошибка: нет БД") and "build_db.py" in e.value.code and "demo.py" in e.value.code


def test_connect_existing(tmp_path):
    db = tmp_path / "attrex.duckdb"
    duckdb.connect(str(db)).close()
    connect(db).close()


def test_require_raw(tmp_path):
    with pytest.raises(DbError) as e:
        require_raw(tmp_path)
    assert "skus.pkl" in e.value.code
    for name in ("sku_desc.parquet", "query_clicks.parquet"):
        (tmp_path / name).touch()
    require_raw(tmp_path, need_feed=False)      # разбор фида уже в кэше: skus.pkl не нужен
