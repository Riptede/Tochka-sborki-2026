"""Решения модератора (b5): заполненная очередь → хранилище решений.

decision: accept — значение верно; fix — верно fixed_value; absent — значения у товара нет или в тексте его нет.
Хранилище — data/processed/review/decisions.parquet: писать в data/attrex.duckdb могут только build_db и
run_pipeline (CLAUDE.md, один писатель DuckDB), поэтому таблицу rev_decisions заполняем отсюда только по флагу
--to-db, когда БД никто не держит. Последнее решение по паре (sku_id, attr_id) главнее прежних.

    uv run python -m attrex.review.decisions --run-id <run_id> --file outputs/<run_id>/review_queue.csv --reviewer anna
"""
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

DECISIONS = {"accept", "fix", "absent"}
COLUMNS = ["sku_id", "attr_id", "run_id", "decision", "value", "value_raw", "reviewer", "decided_at"]


def read_filled_queue(path: Path, run_id: str, reviewer: str, now: datetime | None = None) -> pd.DataFrame:
    """Строки очереди с решением → решения. Неизвестное решение или fix без значения — ошибка, а не тихий пропуск."""
    q = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    q = q[q.decision.str.strip() != ""].copy()
    q["decision"] = q.decision.str.strip().str.lower()
    bad = q[~q.decision.isin(DECISIONS) | ((q.decision == "fix") & (q.fixed_value.str.strip() == ""))]
    if len(bad):
        raise ValueError(f"строки очереди с ошибкой в решении (rank): {bad['rank'].tolist()[:20]}")
    value = [v if d == "accept" else (f.strip() if d == "fix" else None)
             for d, v, f in zip(q.decision, q.value, q.fixed_value)]
    return pd.DataFrame({
        "sku_id": q.sku_id.astype("int64").to_numpy(), "attr_id": q.attr_id.to_numpy(), "run_id": run_id,
        "decision": q.decision.to_numpy(), "value": pd.Series(value, dtype=object), "value_raw": q.value_raw.to_numpy(),
        "reviewer": reviewer,
        "decided_at": pd.Timestamp(now or datetime.now()).floor("s"),
    })[COLUMNS]


class DecisionStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_parquet(self.path)

    def append(self, decisions: pd.DataFrame) -> pd.DataFrame:
        """Добавить решения; по каждой паре (sku_id, attr_id) остаётся последнее."""
        frames = [df for df in (self.load(), decisions) if len(df)]
        merged = pd.concat(frames, ignore_index=True) if frames else decisions
        merged = merged.sort_values("decided_at").drop_duplicates(["sku_id", "attr_id"], keep="last")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(self.path)
        return merged


def store_to_db(db_path: Path, decisions: pd.DataFrame) -> None:
    """rev_decisions = хранилище решений (полная замена). Только когда БД не открыта другими процессами."""
    con = duckdb.connect(str(db_path))
    try:
        con.register("decisions_df", decisions)
        con.execute("BEGIN")
        con.execute("DELETE FROM rev_decisions")
        con.execute("INSERT INTO rev_decisions SELECT sku_id, attr_id, run_id, decision, value, reviewer, decided_at "
                    "FROM decisions_df")
        con.execute("COMMIT")
    finally:
        con.close()


def main() -> None:
    import argparse

    from attrex import paths

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--file", type=Path, required=True, help="заполненная очередь review_queue.csv")
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--store", type=Path, default=paths.PROCESSED / "review" / "decisions.parquet")
    ap.add_argument("--to-db", action="store_true", help="переписать rev_decisions в БД (БД не должна быть занята)")
    ap.add_argument("--db", type=Path, default=paths.DB)
    a = ap.parse_args()
    new = read_filled_queue(a.file, a.run_id, a.reviewer)
    merged = DecisionStore(a.store).append(new)
    print(f"решений добавлено: {len(new):,} ({new.decision.value_counts().to_dict()}); всего в {a.store}: {len(merged):,}")
    if a.to_db:
        store_to_db(a.db, merged)
        print("rev_decisions обновлена")


if __name__ == "__main__":
    main()
