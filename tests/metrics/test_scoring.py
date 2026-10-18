import math
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from attrex.metrics.report import RunInfo, md_table, render
from attrex.metrics.scoring import (
    category_accuracy,
    category_table,
    coverage,
    evaluate,
    macro_f1,
    prf,
    step_contribution,
    traffic_f1,
)

SCHEMA_SQL = Path(__file__).parents[2] / "src" / "attrex" / "db" / "schema.sql"


def outcomes(rows):
    """rows: (category_id, attr_id, status, judged, correct, weight, step[, in_text, fold])."""
    rows = [(*r, *(True, "test")[len(r) - 7:]) for r in rows]
    df = pd.DataFrame(rows, columns=["category_id", "attr_id", "status", "judged", "correct", "weight", "step",
                                     "in_text", "fold"])
    df["in_text"] &= df.judged
    df["accepted"] = df.status.eq("accepted")
    df["has_value"] = df.status.ne("empty")
    df["category_name"] = df.category_id.map({1: "Смартфоны", 2: "Ноутбуки", 3: "Матрасы"})
    return df


DF = outcomes([
    (1, "brand", "accepted", True, True, 0.5, "rules"),
    (1, "brand", "accepted", True, False, 0.5, "rules"),
    (1, "brand", "review", True, True, 0.5, "parse"),
    (1, "brand", "empty", True, False, 0.5, "none", False),   # значения нет в тексте: в R_text не входит
    (1, "color", "accepted", False, False, 0.0, "rules"),     # эталона нет: только в покрытие
    (2, "color", "accepted", True, True, 0.1, "parse"),
])


def test_prf_counts_only_pairs_with_gold():
    m = prf(DF)
    assert m["gold"] == 5 and m["pairs"] == 6
    assert m["P"] == pytest.approx(2 / 3)                      # TP 2, FP 1
    assert m["R"] == pytest.approx(2 / 5)
    assert m["P_any"] == pytest.approx(3 / 4)                  # review тоже в счёт
    assert m["accepted"] == pytest.approx(4 / 6)
    assert m["gold_text"] == 4 and m["text_share"] == pytest.approx(4 / 5)
    assert m["R_text"] == pytest.approx(2 / 4)                 # те же TP, но только на парах со значением в тексте


def test_macro_and_traffic():
    assert macro_f1(DF, min_gold=1) == pytest.approx((prf(DF[DF.category_id == 1])["F1"] + 1.0) / 2)
    brand_f1 = prf(DF[(DF.category_id == 1) & (DF.attr_id == "brand")])["F1"]
    assert traffic_f1(DF) == pytest.approx((0.5 * brand_f1 + 0.1 * 1.0) / 0.6)
    assert math.isnan(macro_f1(DF, min_gold=100))


def test_category_table_top_tail_and_holdout():
    df = pd.concat([DF, outcomes([(3, "brand", "accepted", True, True, 0.2, "rules", True, "holdout_category")])])
    t = category_table(df, top=1)
    assert list(t.index) == ["Смартфоны (1)", "хвост: 1 категорий", "holdout_category: 1 категорий"]
    assert t.loc["Смартфоны (1)", "traffic"] == pytest.approx(0.5)       # brand 0.5 + color 0.0
    assert t.loc["Смартфоны (1)", "gold"] == 4 and t.loc["хвост: 1 категорий", "F1"] == pytest.approx(1.0)
    assert t.loc["holdout_category: 1 категорий", "traffic"] == pytest.approx(0.2)


def test_category_accuracy_by_card():
    df = pd.DataFrame({"sku_id": [1, 1, 2, 3, 4], "category_id": [10, 10, 11, None, 5],
                       "true_category_id": [10, 10, 12, 13, None]})
    acc = category_accuracy(df)
    assert acc["cards"] == 3                                   # карточка без категории в БД не в счёт
    assert acc["coverage"] == pytest.approx(2 / 3)
    assert acc["accuracy"] == pytest.approx(1 / 3) and acc["accuracy_covered"] == pytest.approx(1 / 2)


def test_step_contribution_and_table():
    steps = step_contribution(DF)
    assert steps.loc["rules", "rows"] == 3 and steps.loc["rules", "P"] == pytest.approx(0.5)
    table = md_table(steps, ["rows", "share", "P"], "step")
    assert table.splitlines()[0] == "| step | rows | share | P |"
    assert "| rules | 3 | 75.0% | 50.0% |" in table


def test_coverage_before_and_after():
    con = duckdb.connect()
    con.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    con.executemany("INSERT INTO core_cards (sku_id, title, has_offer) VALUES (?, ?, ?)",
                    [(1, "оффер", True), (2, "без оффера", False)])
    con.executemany("INSERT INTO core_gold (sku_id, attr_id, value, is_placeholder) VALUES (?, ?, ?, ?)",
                    [(1, "brand", "samsung", False), (1, "color", None, True)])     # цвет — заглушка: не заполнен
    rows = [(1, "brand", "accepted"), (1, "color", "accepted"), (2, "brand", "review"), (2, "color", "empty")]
    con.executemany("INSERT INTO pred_predictions (run_id, sku_id, attr_id, status) VALUES ('r', ?, ?, ?)", rows)
    cov = coverage(con, "r")
    assert list(cov.index) == ["офферы", "карточки без оффера", "все карточки"]
    offers, total = cov.loc["офферы"], cov.loc["все карточки"]
    assert (offers.pairs, offers.feed, offers.accepted, offers.feed_or_accepted) == (2, 0.5, 1.0, 1.0)
    assert (total.cards, total.pairs, total.feed, total.accepted, total.any_value) == (2, 4, 0.25, 0.5, 0.75)


def test_report_has_cost_llm_coverage_and_categories():
    m = evaluate(DF.assign(sku_id=range(len(DF)), true_category_id=DF.category_id, value="x", seller_seg="1P",
                           seller_size="нет", is_refurb=False, category_source="feed"))
    m.coverage = pd.DataFrame({"cards": [1], "pairs": [2], "feed": [0.5], "accepted": [1.0], "feed_or_accepted": [1.0],
                               "any_value": [1.0]}, index=["все карточки"])
    usage = {"llm_calls": 10, "llm_cost_rub": 0.5, "llm_cards": 8, "llm_values": 20, "llm_ungrounded": 5,
             "llm_invalid": 1}
    report = render(RunInfo("r", ["rules", "llm"], ["valid"], n_cards=100, cost_rub=0.5, usage=usage), m)
    assert "Стоимость: 0.50 ₽ (5.00 ₽ на 1 000 карточек)." in report
    assert "## LLM-шаг: расход и проверка ответов" in report and "| из них отброшено проверкой цитаты | 5 (25.0%) |" in report
    assert "## Покрытие каталога: до и после" in report and "## По категориям" in report
    assert "|  | P | R | F1 | R_text | macro_F1 |" in report
