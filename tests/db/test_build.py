import duckdb
import pytest

from attrex.db.build import BuildConfig, build


@pytest.fixture
def db(tmp_path, raw_dir):
    cfg = BuildConfig(raw_dir=raw_dir, interim_dir=tmp_path / "interim", db_path=tmp_path / "attrex.duckdb",
                      memory_limit="512MB", threads=1)
    counts = build(cfg, log=lambda _: None)
    assert not (tmp_path / "attrex.duckdb.tmp").exists()
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    yield con, counts
    con.close()


def gold(con, sku, attr):
    return con.execute("SELECT value, unit, is_placeholder FROM core_gold WHERE sku_id = ? AND attr_id = ? "
                       "ORDER BY value", [sku, attr]).fetchall()


def test_counts(db):
    _, counts = db
    assert counts["raw_offers"] == 3
    assert "kinds_df" not in counts and "raw_descriptions" not in counts
    assert counts["core_cards"] == 5            # 4 описания + оффер без описания
    assert counts["core_splits"] == 3           # только карточки с эталоном


def test_gold_numbers_in_canonical_units(db):
    con, _ = db
    assert gold(con, 30000001, "storage") == [("256", "GB", False)]
    assert gold(con, 400000002, "storage") == [("1024", "GB", False)]   # 1 ТБ, имя поля с пробелом в начале
    assert gold(con, 30000001, "battery") == [("4000", "mAh", False)]   # точное _search, а не корзина фасета
    assert gold(con, 30000001, "screen_diag") == [("6.2", "in", False)]


def test_gold_categories_and_placeholders(db):
    con, _ = db
    assert gold(con, 30000001, "color") == [("черный", None, False)]   # фасет в каноне
    assert gold(con, 400000002, "color") == [("мятный", None, False)]  # фасет-заглушка → сырое значение
    assert gold(con, 400000003, "color") == [(None, None, True)]       # только заглушка
    assert gold(con, 30000001, "brand") == [("samsung", None, False)]
    assert gold(con, 30000001, "model") == [("galaxy s25", None, False)]   # без цвета
    assert gold(con, 400000002, "model") == [("redmi 13", None, False)]
    assert gold(con, 400000003, "model") == [(None, None, True)]           # модель из одного цвета


def test_cards(db):
    con, _ = db
    rows = {r[0]: r[1:] for r in con.execute(
        "SELECT sku_id, has_offer, category_id, category_source, seller_seg, is_refurb, clicks, description "
        "FROM core_cards").fetchall()}
    assert rows[30000001][:6] == (True, 761, "feed", "1P", False, 2)
    assert rows[400000002][3] == "3P"
    assert rows[500000005][:3] == (False, 900, "clicks")
    assert rows[500000004][4] is True
    assert rows[400000003][0] is True and rows[400000003][6] is None   # оффер без описания
    desc = rows[30000001][6]
    assert "<" not in desc and "Лучшие товары" not in desc and '6,2"' in desc and "\n\n\n" not in desc
    assert rows[400000002][6] == "Цвет: мятный\nБренд «Xiaomi»"   # теги и сущности закодированы дважды


def test_schema_inherits_and_keeps_required(db):
    con, _ = db
    rows = con.execute("SELECT attr_id, required, inherited_from FROM core_category_schema "
                       "WHERE category_id = 761").fetchall()
    req = {a for a, r, _ in rows if r}
    assert req == {"brand", "model", "storage", "screen_diag", "battery", "color"}   # список ТЗ, не правило 70%
    assert {inh for _, _, inh in rows} == {1}   # 2 оффера < 50: схема родителя


def test_ref_values_and_splits(db):
    con, _ = db
    colors = dict(con.execute("SELECT value, synonyms FROM core_ref_values WHERE attr_id = 'color'").fetchall())
    assert "черный" in colors and "мятный" in colors
    folds = {f for (f,) in con.execute("SELECT fold FROM core_splits").fetchall()}
    assert folds <= {"train", "valid", "test", "holdout_category"}
