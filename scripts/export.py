"""Файл сдачи (b4): outputs/predictions.parquet и .csv из прогона pred_predictions.

    uv run python scripts/export.py --run-id 20261018-0400-rules-parse

Одна строка на пару (sku_id, attr_id) для всех обязательных атрибутов категории карточки, включая пустые;
колонки — PREDICTION_COLUMNS (contracts.py). Для сдачи берите прогон по всем карточкам (без --fold и --limit).
"""
import argparse
from pathlib import Path

from attrex import paths
from attrex.db import connect
from attrex.runs import export


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--db", type=Path, default=paths.DB)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS)
    a = ap.parse_args()
    con = connect(a.db)
    try:
        for p in export(con, a.run_id, a.out):
            print(f"{p} ({p.stat().st_size / 2**20:,.1f} МБ)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
