"""Маленький синтетический набор в формате файлов организатора: фид в pickle, описания и клики в parquet."""
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

CATEGORIES = [
    {"@id": "1", "#text": "Электроника"},
    {"@id": "761", "@parentId": "1", "#text": "Смартфоны"},
    {"@id": "900", "@parentId": "1", "#text": "Чехлы"},
]

OFFERS = [
    {"@id": "30000001", "@available": "true", "vendor": "Samsung", "model": "Galaxy S25 черный",
     "name": "Смартфон Samsung Galaxy S25 8/256GB черный", "categories": {"categoryId": "761"},
     "param": [
         {"@name": "Продавец", "#text": "М.Видео"},
         {"@name": "Подборки", "#text": "игровые"},
         {"@name": "Цвет", "#text": "Черный"},
         {"@name": "Цвет_search", "#text": "черный"},
         {"@name": "Встроенная память", "#text": "256 ГБ"},
         {"@name": "Встроенная память (ROM)_search", "@unit": "ГБ", "#text": "256"},
         {"@name": "Емкость аккумулятора", "#text": "3000-5000 мАч"},
         {"@name": "Емкость аккумулятора_search", "@unit": "мАч", "#text": "4000"},
         {"@name": "Диагональ_search", "@unit": "\"", "#text": "6.2"},
     ]},
    {"@id": "400000002", "@available": "true", "vendor": "Xiaomi", "model": "Redmi 13 green",
     "name": "Смартфон Xiaomi Redmi 13 8/1TB", "categories": {"categoryId": "761"},
     "param": [
         {"@name": "Продавец", "#text": "ООО Ромашка"},
         {"@name": "Цвет", "#text": "другие цвета"},
         {"@name": "Цвет_search", "#text": "мятный"},
         {"@name": " Встроенная память (ROM)_search", "@unit": "ТБ", "#text": "1"},
     ]},
    {"@id": "400000003", "@available": "true", "vendor": "UNIMALL", "model": "Чехол для Honor",
     "name": "Чехол UNIMALL для Honor", "categories": {"categoryId": "900"},
     "param": {"@name": "Цвет_search", "#text": "другой"}},
]

FEED = {"yml_catalog": {"@date": "2026-05-13", "shop": {
    "categories": {"category": CATEGORIES}, "offers": {"offer": OFFERS}}}}

DESCRIPTIONS = [
    (30000001, "Смартфон Samsung Galaxy S25 8/256GB черный",
     "Экран <b>6,2&quot;</b><br>Батарея 4000 мА*ч<br><br><br>Лучшие товары из Китая"),
    (400000002, "Смартфон Xiaomi Redmi 13 8/1TB", "Цвет: мятный"),
    (500000004, "Восстановленный смартфон Apple iPhone 13 128GB, хороший", "Внешний вид: «Хороший»."),
    (500000005, "Чехол для смартфона GSS -", "Характеристики:\n• Цвет: black"),
]

CLICKS = [(500000005, 900)] * 3 + [(30000001, 761)] * 2


@pytest.fixture
def raw_dir(tmp_path, protocol=4):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "skus.pkl").write_bytes(pickle.dumps(FEED, protocol=protocol))
    pq.write_table(pa.table({
        "sku_id": pa.array([d[0] for d in DESCRIPTIONS], pa.int64()),
        "title": [d[1] for d in DESCRIPTIONS],
        "description": [d[2] for d in DESCRIPTIONS],
    }), raw / "sku_desc.parquet")
    pq.write_table(pa.table({
        "sku_id": pa.array([c[0] for c in CLICKS], pa.uint64()),
        "sku_seo_id": pa.array([c[1] for c in CLICKS], pa.uint64()),
    }), raw / "query_clicks.parquet")
    return raw
