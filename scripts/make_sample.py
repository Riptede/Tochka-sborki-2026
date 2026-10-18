"""Выборка реальных данных для демо (b1): sample/raw/ в формате файлов организатора.

    uv run python scripts/make_sample.py      # нужны data/raw/*, data/interim/ (кэш разбора фида) и data/attrex.duckdb

Офферы с описанием длиннее 100 символов — из семи категорий, в каждой не меньше 50 офферов: на выборке схемы обязательных атрибутов строятся
так же, как на полных данных. Карточки без оффера — с блоком «ключ: значение» в описании и с категорией из кликов:
на них видны шаг 0 (категория по заголовку) и парсер. Отбор детерминированный (по md5 от sku_id), повторный запуск
даёт те же файлы. skus.pkl собирается из кэша разбора фида в той же структуре xmltodict, что у организатора.
"""
import argparse
import pickle
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from attrex import paths

# категория → сколько офферов взять
OFFER_CATEGORIES = {
    761: 150,     # Смартфоны: обязательные поля из ТЗ
    987: 120,     # Ноутбуки: обязательные поля из ТЗ
    914: 60,      # iPhone
    1682: 60,     # Телевизоры
    36480: 60,    # Матрасы: самая большая категория каталога
    1189: 60,     # Чехлы для смартфонов
    30446: 60,    # Смесители
}
# начало заголовка карточки без оффера → сколько карточек взять
NO_OFFER_TITLES = {"Смартфон": 40, "Ноутбук": 40, "Телевизор": 30, "Матрас": 30, "Чехол": 30, "Смеситель": 30}
CLICKS_CARDS = 40          # карточки без оффера, категория которых известна по кликам
CLICKS_ROWS_PER_SKU = 20   # строк кликов на карточку: для весов по трафику хватает


def pick_skus(con: duckdb.DuckDBPyConnection) -> tuple[list[int], list[int]]:
    offers = []
    for cid, n in OFFER_CATEGORIES.items():
        offers += [r[0] for r in con.execute(
            "SELECT sku_id FROM core_cards WHERE has_offer AND category_id = ? AND length(description) > 100 "
            "ORDER BY md5(sku_id::VARCHAR) LIMIT ?",
            [cid, n]).fetchall()]
    cards = []
    for prefix, n in NO_OFFER_TITLES.items():
        # описание с несколькими строками «ключ: значение» — материал для парсера
        cards += [r[0] for r in con.execute(r"""
            SELECT sku_id FROM core_cards
            WHERE NOT has_offer AND category_source IS NULL AND title ILIKE ? || ' %'
              AND len(regexp_extract_all(description, '\n[^\n:]{2,40}:[ \t]*\S')) >= 3
            ORDER BY md5(sku_id::VARCHAR) LIMIT ?""", [prefix, n]).fetchall()]
    cards += [r[0] for r in con.execute("""
        SELECT sku_id FROM core_cards
        WHERE NOT has_offer AND category_source = 'clicks' AND category_id IN (SELECT unnest(?))
        ORDER BY md5(sku_id::VARCHAR) LIMIT ?""", [list(OFFER_CATEGORIES), CLICKS_CARDS]).fetchall()]
    return offers, cards


def build_feed(interim: Path, offer_ids: list[int]) -> dict:
    """Офферы из кэша разбора фида обратно в структуру xmltodict: порядок параметров сохраняется."""
    ids = [str(i) for i in offer_ids]
    offers = pq.read_table(interim / "offers.parquet", filters=[("offer_id", "in", ids)])
    params = pq.read_table(interim / "params.parquet", filters=[("offer_id", "in", ids)])
    by_offer: dict[str, list[dict]] = {}
    for oid, name, value, unit in zip(*(params.column(c).to_pylist() for c in ("offer_id", "param_name", "value",
                                                                               "unit"))):
        p = {"@name": name}
        if unit is not None:
            p["@unit"] = unit
        if value is not None:
            p["#text"] = value
        by_offer.setdefault(oid, []).append(p)
    out = []
    for o in offers.to_pylist():
        d = {"@id": o["offer_id"], "@available": o["available"]}
        for k in ("vendor", "model", "name"):
            if o[k] is not None:
                d[k] = o[k]
        if o["category_id"] is not None:
            d["categories"] = {"categoryId": o["category_id"]}
        if o["offer_id"] in by_offer:
            d["param"] = by_offer[o["offer_id"]]
        out.append(d)
    cats = []
    for c in pq.read_table(interim / "categories.parquet").to_pylist():
        d = {"@id": c["id"]}
        if c["parent_id"] is not None:
            d["@parentId"] = c["parent_id"]
        d["#text"] = c["name"]
        cats.append(d)
    return {"yml_catalog": {"shop": {"categories": {"category": cats}, "offers": {"offer": out}}}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=paths.RAW)
    ap.add_argument("--interim", type=Path, default=paths.INTERIM)
    ap.add_argument("--db", type=Path, default=paths.DB)
    ap.add_argument("--out", type=Path, default=paths.ROOT / "sample" / "raw")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(a.db), read_only=True)
    try:
        offers, cards = pick_skus(con)
    finally:
        con.close()
    skus = offers + cards

    feed = build_feed(a.interim, offers)
    (a.out / "skus.pkl").write_bytes(pickle.dumps(feed, protocol=4))

    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE pick AS SELECT unnest(?::BIGINT[]) AS sku_id", [skus])
    desc = con.execute(f"""
        SELECT d.* FROM read_parquet('{(a.raw / 'sku_desc.parquet').as_posix()}') d SEMI JOIN pick USING (sku_id)
        ORDER BY sku_id""").to_arrow_table()
    pq.write_table(desc.cast(pq.read_schema(a.raw / "sku_desc.parquet")), a.out / "sku_desc.parquet",
                   compression="zstd")
    clicks_schema = pq.read_schema(a.raw / "query_clicks.parquet")
    clicks = con.execute(f"""
        SELECT * EXCLUDE (rn) FROM (
            SELECT q.*, row_number() OVER (PARTITION BY q.sku_id ORDER BY md5(q."toValidUTF8(query_text)")) AS rn
            FROM read_parquet('{(a.raw / 'query_clicks.parquet').as_posix()}') q
            WHERE q.sku_id IN (SELECT sku_id FROM pick))
        WHERE rn <= {CLICKS_ROWS_PER_SKU} ORDER BY sku_id, "toValidUTF8(query_text)" """).to_arrow_table()
    pq.write_table(clicks.cast(clicks_schema), a.out / "query_clicks.parquet", compression="zstd")

    print(f"офферов: {len(feed['yml_catalog']['shop']['offers']['offer'])}, карточек без оффера: {len(cards)}, "
          f"описаний: {desc.num_rows}, строк кликов: {clicks.num_rows}")
    for f in sorted(a.out.iterdir()):
        print(f"  {f.name:22} {f.stat().st_size / 1024:8,.0f} КБ")


if __name__ == "__main__":
    main()
