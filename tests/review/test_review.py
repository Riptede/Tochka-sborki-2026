"""Очередь, решения и обратная связь — на маленькой БД в памяти со схемой проекта."""
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from attrex.registry import load_registry
from attrex.review.decisions import DecisionStore, read_filled_queue
from attrex.review.feedback import agreement, synonyms_from_decisions
from attrex.review.queue import QUEUE_COLUMNS, build_queue, write_queue

SCHEMA_SQL = Path(__file__).parents[2] / "src" / "attrex" / "db" / "schema.sql"
RUN = "20261018-0300-test"


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    con.executemany("INSERT INTO core_cards (sku_id, title, clicks, category_id) VALUES (?, ?, ?, ?)",
                    [(1, "Смартфон A", 1000, 761), (2, "Смартфон B", 0, 761), (3, "Чехол C", 50, 900)])
    con.execute("INSERT INTO core_traffic_weights VALUES (761, 'color', 0.01)")
    rows = [(1, 761, "color", "серый", 0.6, "review", "titanium grey"), (2, 761, "color", "белый", 0.6, "review", "white"),
            (3, 900, "color", "черный", 0.2, "review", "black"), (1, 761, "brand", "samsung", 0.99, "accepted", "Samsung")]
    con.executemany("INSERT INTO pred_predictions (run_id, sku_id, category_id, attr_id, value, confidence, status, "
                    "step, source, value_raw, evidence) VALUES (?, ?, ?, ?, ?, ?, ?, 'rules', 'title', ?, ?)",
                    [(RUN, *r, r[-1]) for r in rows])
    yield con
    con.close()


def test_queue_only_review_sorted_by_priority(con):
    q = build_queue(con, RUN)
    assert list(q.columns) == QUEUE_COLUMNS
    # 1-я: клики и вес трафика; 2-я: вес трафика без кликов; 3-я: уверенность ниже, но категория без трафика
    assert q.sku_id.tolist() == [1, 2, 3]
    assert set(q.attr_id) == {"color"} and (q.decision == "").all()
    assert len(build_queue(con, RUN, limit=1)) == 1


def test_decisions_roundtrip_and_feedback(con, tmp_path):
    q = build_queue(con, RUN)
    q.loc[q.sku_id == 1, "decision"] = "accept"
    q.loc[q.sku_id == 2, ["decision", "fixed_value"]] = ["fix", "серебристый"]
    q.loc[q.sku_id == 3, "decision"] = "absent"
    path = write_queue(q, tmp_path / "queue.csv")
    d = read_filled_queue(path, RUN, "anna", now=datetime(2026, 10, 18, 3, 0))
    assert dict(zip(d.sku_id, d.value)) == {1: "серый", 2: "серебристый", 3: None}

    store = DecisionStore(tmp_path / "decisions.parquet")
    store.append(d)
    later = d[d.sku_id == 2].assign(value="белый", decision="accept", decided_at=pd.Timestamp(2026, 10, 18, 4))
    merged = store.append(later)
    assert len(merged) == 3 and merged.set_index("sku_id").loc[2, "value"] == "белый"   # последнее решение главнее

    syn = synonyms_from_decisions(store.load(), load_registry())
    assert syn == {"color": {"titanium grey": "серый", "white": "белый"}}
    agree = agreement(store.load(), con.execute("SELECT sku_id, attr_id, confidence FROM pred_predictions").df())
    assert agree.loc["color", "decisions"] == 3 and agree.loc["color", "accept"] == pytest.approx(2 / 3)


def test_bad_decision_is_an_error(con, tmp_path):
    q = build_queue(con, RUN)
    q.loc[q.sku_id == 1, "decision"] = "fix"            # fix без исправленного значения
    path = write_queue(q, tmp_path / "queue.csv")
    with pytest.raises(ValueError, match="ошибкой"):
        read_filled_queue(path, RUN, "anna")
