"""Интерфейс модератора: страница по очереди и выгрузка решений — JS страницы исполняется в quickjs."""
import json
import re
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from attrex.review.app import page_data, read_queue, render_page, write_page
from attrex.review.decisions import read_filled_queue
from attrex.review.queue import QUEUE_COLUMNS, build_queue, write_queue

quickjs = pytest.importorskip("quickjs")

SCHEMA_SQL = Path(__file__).parents[2] / "src" / "attrex" / "db" / "schema.sql"
RUN = "20261018-0300-test"
TRICKY = 'Смартфон "Galaxy", 8/256GB\nсерый </script>'        # кавычки, запятая, перенос строки, тег


@pytest.fixture
def queue(tmp_path):
    con = duckdb.connect()
    con.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    con.executemany("INSERT INTO core_cards (sku_id, title, description, clicks, category_id) VALUES (?, ?, ?, ?, ?)",
                    [(1, TRICKY, "Корпус цвета Titanium Grey.", 1000, 761), (2, "Смартфон B", "Экран 6,7 дюйма.", 0, 761),
                     (3, "Чехол C", "", 50, 900)])
    con.execute("INSERT INTO raw_categories VALUES (761, NULL, 'Смартфоны')")
    con.executemany("INSERT INTO core_ref_values VALUES ('color', ?, [], ?)", [("серый", 50), ("серебристый", 20)])
    con.execute("INSERT INTO core_traffic_weights VALUES (761, 'color', 0.01)")
    rows = [(1, 761, "color", "серый", 0.6, "titanium grey", "Titanium Grey"),
            (2, 761, "screen_diag", "6.5", 0.5, "6,7 дюйма", "6,7 дюйма"),
            (3, 900, "color", "черный", 0.2, "black", "black")]
    con.executemany("INSERT INTO pred_predictions (run_id, sku_id, category_id, attr_id, value, confidence, status, "
                    "step, source, value_raw, evidence) VALUES (?, ?, ?, ?, ?, ?, 'review', 'rules', 'title', ?, ?)",
                    [(RUN, *r) for r in rows])
    path = write_queue(build_queue(con, RUN), tmp_path / "review_queue.csv")
    yield path, con
    con.close()


@pytest.fixture(scope="module")
def js():
    """Чистые функции страницы (<script id="logic">) в JS-движке — тот же код, что исполняет браузер."""
    ctx = quickjs.Context()
    ctx.eval(re.search(r'<script id="logic">(.*?)</script>', render_page({}), re.DOTALL).group(1))

    def call(fn: str, *args):
        return json.loads(ctx.eval(f"JSON.stringify({fn}(...{json.dumps(args, ensure_ascii=False)}))"))
    return call


def test_page_embeds_queue_texts_and_dictionary(queue, tmp_path):
    path, con = queue
    page = write_page(path, RUN, tmp_path / "review.html", con).read_text(encoding="utf-8")
    block = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.DOTALL).group(1)
    assert "</script>" not in block and "<" not in block                    # текст карточки не закроет блок данных
    data = json.loads(block)
    assert data["columns"] == QUEUE_COLUMNS and [r["sku_id"] for r in data["rows"]] == ["1", "3", "2"]   # приоритет
    assert data["rows"][0]["title"] == TRICKY
    assert data["cards"]["1"] == {"description": "Корпус цвета Titanium Grey.", "category": "Смартфоны"}
    assert data["options"]["color"] == ["серый", "серебристый"]
    assert data["attrs"]["screen_diag"]["unit"] == "in" and 0 < data["attrs"]["color"]["threshold"] <= 1
    no_db = page_data(path, RUN)                                              # без БД: только очередь
    assert no_db["cards"] == {} and not no_db["has_descriptions"] and len(no_db["rows"]) == 3


def test_downloaded_csv_loads_with_read_filled_queue(queue, tmp_path, js):
    path, con = queue
    data = page_data(path, RUN, con)
    fix = js("normalizeFix", data["attrs"]["color"], " Серебристый ", data["options"]["color"])
    assert fix == {"ok": True, "value": "серебристый"}                        # написание — как в справочнике
    decisions = {"1|color": {"decision": "fix", "fixed_value": fix["value"]},
                 "2|screen_diag": {"decision": "accept", "fixed_value": "не выгружается: не fix"},
                 "3|color": {"decision": "absent", "fixed_value": ""}}
    out = tmp_path / "decided.csv"
    out.write_text(js("queueCsv", data["columns"], data["rows"], decisions), encoding="utf-8", newline="")
    d = read_filled_queue(out, RUN, "anna", now=datetime(2026, 10, 18, 3, 0))
    assert dict(zip(d.sku_id, d.decision)) == {1: "fix", 2: "accept", 3: "absent"}
    assert dict(zip(d.sku_id, d.value)) == {1: "серебристый", 2: "6.5", 3: None}
    assert d.set_index("sku_id").loc[1, "value_raw"] == "titanium grey"

    out.write_text(js("queueCsv", data["columns"], data["rows"], {"3|color": decisions["3|color"]}),
                   encoding="utf-8", newline="")
    assert read_filled_queue(out, RUN, "anna").sku_id.tolist() == [3]          # нерешённые строки не загружаются


def test_downloaded_csv_keeps_queue_fields(queue, tmp_path, js):
    path, con = queue
    data = page_data(path, RUN, con)
    out = tmp_path / "decided.csv"
    out.write_text(js("queueCsv", data["columns"], data["rows"], {}), encoding="utf-8", newline="")
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")                       # BOM: Excel откроет кириллицу
    assert read_queue(out).to_dict(orient="records") == data["rows"]           # заголовок с "," и \n — как был


@pytest.mark.parametrize("attr, raw, options, expected", [
    ({"kind": "number", "unit": "in"}, "6,7", None, "6.7"),
    ({"kind": "number", "unit": "GB"}, "256 ГБ", None, None),                  # единица и пояснения — нельзя
    ({"kind": "category", "unit": None}, "Серый", ["серый"], "серый"),
    ({"kind": "category", "unit": None}, "лиловый", ["серый"], None),          # нет в справочнике
    ({"kind": "text", "unit": None}, "  galaxy   s25 ", None, "galaxy s25"),
    ({"kind": "text", "unit": None}, "   ", None, None),
])
def test_fix_value_follows_protocol(js, attr, raw, options, expected):
    res = js("normalizeFix", attr, raw, options)
    assert (res["value"] if res["ok"] else None) == expected


def test_quote_is_found_ignoring_case_spaces_and_yo(js):
    text = 'Корпус  ЧЁРНОГО цвета, экран 6.7"'
    start, end = js("quoteRange", text, "черного   цвета")
    assert text[start:end] == "ЧЁРНОГО цвета"
    assert js("quoteRange", text, '6.7"') == [len(text) - 4, len(text)]
    assert js("quoteRange", text, "белого") is None


def test_page_scripts_compile():
    """Синтаксическая ошибка в скрипте интерфейса не должна дойти до модератора: компилируем, не исполняя."""
    html = render_page({})
    scripts = re.findall(r"<script(?: id=\"logic\")?>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 2
    ctx = quickjs.Context()
    for src in scripts:
        ctx.eval("(function () {" + src + "\n})")
