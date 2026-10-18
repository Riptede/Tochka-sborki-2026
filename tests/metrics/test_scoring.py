import math

import pandas as pd
import pytest

from attrex.metrics.report import md_table
from attrex.metrics.scoring import category_accuracy, macro_f1, prf, step_contribution, traffic_f1


def outcomes(rows):
    """rows: (category_id, attr_id, status, judged, correct, weight, step)."""
    df = pd.DataFrame(rows, columns=["category_id", "attr_id", "status", "judged", "correct", "weight", "step"])
    df["accepted"] = df.status.eq("accepted")
    df["has_value"] = df.status.ne("empty")
    return df


DF = outcomes([
    (1, "brand", "accepted", True, True, 0.5, "rules"),
    (1, "brand", "accepted", True, False, 0.5, "rules"),
    (1, "brand", "review", True, True, 0.5, "parse"),
    (1, "brand", "empty", True, False, 0.5, "none"),
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


def test_macro_and_traffic():
    assert macro_f1(DF, min_gold=1) == pytest.approx((prf(DF[DF.category_id == 1])["F1"] + 1.0) / 2)
    brand_f1 = prf(DF[(DF.category_id == 1) & (DF.attr_id == "brand")])["F1"]
    assert traffic_f1(DF) == pytest.approx((0.5 * brand_f1 + 0.1 * 1.0) / 0.6)
    assert math.isnan(macro_f1(DF, min_gold=100))


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
