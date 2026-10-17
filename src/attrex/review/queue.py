"""Очередь ручной проверки (b5): значения со статусом review из прогона, самые важные — первыми.

Приоритет — где ошибка стоит дороже и где модель меньше уверена:
    priority = (1 + log(1 + клики карточки)) · (1 + 1000 · вес пары (категория, атрибут) по трафику) · (1 − уверенность)
Вес по трафику — core_traffic_weights (b1), клики — core_cards. Очередь — CSV для модератора (открывается в Excel):
колонки decision (accept | fix | absent) и fixed_value он заполняет сам, затем решения загружает decisions.py.

    uv run python -m attrex.review.queue --run-id 20261018-0300-calibrated-valid --limit 500
"""
from pathlib import Path

import duckdb
import pandas as pd

QUEUE_COLUMNS = ["rank", "sku_id", "attr_id", "value", "confidence", "step", "source", "value_raw", "evidence",
                 "title", "category_id", "clicks", "priority", "decision", "fixed_value"]

_QUEUE_SQL = """
    SELECT p.sku_id, p.attr_id, p.value, p.confidence, p.step, p.source, p.value_raw, p.evidence,
           c.title, p.category_id, coalesce(c.clicks, 0) AS clicks,
           (1 + ln(1 + coalesce(c.clicks, 0))) * (1 + 1000 * coalesce(w.weight, 0))
               * (1 - coalesce(p.confidence, 0)) AS priority
    FROM pred_predictions p
    JOIN core_cards c USING (sku_id)
    LEFT JOIN core_traffic_weights w ON w.category_id = p.category_id AND w.attr_id = p.attr_id
    WHERE p.run_id = ? AND p.status = 'review'
    ORDER BY priority DESC, p.sku_id, p.attr_id
"""


def build_queue(con: duckdb.DuckDBPyConnection, run_id: str, limit: int | None = None) -> pd.DataFrame:
    sql = _QUEUE_SQL + (f" LIMIT {int(limit)}" if limit else "")
    df = con.execute(sql, [run_id]).df()
    df.insert(0, "rank", range(1, len(df) + 1))
    df["decision"] = ""
    df["fixed_value"] = ""
    return df[QUEUE_COLUMNS]


def write_queue(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")      # utf-8-sig: Excel открывает кириллицу без танцев
    return path


def main() -> None:
    import argparse

    from attrex import paths

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=500, help="сколько значений в очереди")
    ap.add_argument("--db", type=Path, default=paths.DB)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS)
    a = ap.parse_args()
    con = duckdb.connect(str(a.db), read_only=True)
    try:
        df = build_queue(con, a.run_id, a.limit)
    finally:
        con.close()
    path = write_queue(df, a.out / a.run_id / "review_queue.csv")
    print(f"{path}: {len(df):,} значений; по атрибутам: {df.attr_id.value_counts().head(8).to_dict()}")


if __name__ == "__main__":
    main()
