"""Метрики и ablation по run_id (b4) → outputs/<run_id>/report.md.

    uv run python scripts/evaluate.py --run-id 20261018-0300-rules-parse-valid
    uv run python scripts/evaluate.py --run-id <rules+parse> --compare <rules>      # ablation: правила → + парсер
    uv run python scripts/evaluate.py --run-id <итог> --folds test,holdout_category # итоговые метрики

Словари, пороги и калибровку подбираем на train/valid; test и holdout_category — только для итоговых цифр (CLAUDE.md).
"""
import argparse
import json
from pathlib import Path

import duckdb

from attrex import paths
from attrex.db import connect
from attrex.metrics.report import RunInfo, render
from attrex.metrics.scoring import comparison_same, coverage, evaluate, load_outcomes


def run_info(con: duckdb.DuckDBPyConnection, run_id: str, folds: list[str]) -> RunInfo:
    row = con.execute("SELECT config, coalesce(llm_cost_rub, 0) + coalesce(embed_cost_rub, 0) FROM pred_runs "
                      "WHERE run_id = ?", [run_id]).fetchone()
    if row is None:
        raise SystemExit(f"нет прогона {run_id} в pred_runs")
    cfg = json.loads(row[0])
    return RunInfo(run_id, cfg.get("steps", []), folds, cfg.get("n_cards"), float(row[1]), usage=cfg.get("usage", {}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--compare", default="", help="прогоны для ablation через запятую (в порядке добавления шагов)")
    ap.add_argument("--folds", default="valid", help="фолды оценки через запятую")
    ap.add_argument("--db", type=Path, default=paths.DB)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS)
    a = ap.parse_args()
    folds = [f.strip() for f in a.folds.split(",") if f.strip()]
    same = comparison_same()

    con = connect(a.db)
    try:
        def measure(run_id: str):
            return run_info(con, run_id, folds), evaluate(load_outcomes(con, folds, same, run_id=run_id))

        info, metrics = measure(a.run_id)
        metrics.coverage = coverage(con, a.run_id)
        compare = [r.strip() for r in a.compare.split(",") if r.strip()]
        ablation = [measure(r) for r in compare] + [(info, metrics)] if compare else []
    finally:
        con.close()

    out = a.out / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render(info, metrics, ablation), encoding="utf-8")
    h = metrics.headline()
    print(f"{a.run_id} на {', '.join(folds)}: " + ", ".join(f"{k} {v:.1%}" for k, v in h.items()))
    print(f"отчёт: {out / 'report.md'}")


if __name__ == "__main__":
    main()
