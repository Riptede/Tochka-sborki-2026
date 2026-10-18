"""Итоговые результаты для сдачи: прогоны, метрики и файл предсказаний → results/ (в git).

    uv run python scripts/make_results.py                                   # всё заново: ≈1 ч на 3 процессах
    uv run python scripts/make_results.py --full-run 20261018-1442-full-catalog   # полный прогон уже есть в БД

Что запускается (все прогоны — обычными scripts/run_pipeline.py и scripts/evaluate.py):
1. полный каскад по всем карточкам каталога — файл сдачи;
2. ablation на test и holdout_category: правила → + парсер → + retrieval → весь каскад; «до калибровки» —
   весь каскад с сырой уверенностью;
3. шаг 0: весь каскад на test и holdout_category со скрытой категорией.
Словари строятся только по train, калибровка — по valid; test и holdout_category в подборе не участвуют.

В results/ попадает: <полный прогон>/predictions.parquet, report.md (итог на test+holdout_category с ablation),
run.json (шаги, конфиг, коммит), отчёты «до калибровки» и шага 0.
"""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from attrex import paths

FINAL_FOLDS = "test,holdout_category"
ABLATION = [("rules", "rules"), ("rules-parse", "rules,parse"), ("rules-parse-retrieval", "rules,parse,retrieval")]
RESULTS = paths.ROOT / "results"


def sh(script: str, *args: str) -> None:
    cmd = [sys.executable, str(paths.ROOT / "scripts" / script), *args]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=paths.ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-run", help="готовый прогон по всему каталогу (иначе запускается заново)")
    ap.add_argument("--workers", default="3")
    ap.add_argument("--db", type=Path, default=paths.DB)
    a = ap.parse_args()
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    db = ["--db", str(a.db)]
    common = [*db, "--workers", a.workers]

    full = a.full_run or f"{stamp}-full-catalog"
    if not a.full_run:
        sh("run_pipeline.py", *common, "--run-id", full)

    ablation = []
    for label, steps in ABLATION:
        rid = f"{stamp}-{label}-final"
        sh("run_pipeline.py", *common, "--steps", steps, "--fold", FINAL_FOLDS, "--run-id", rid)
        ablation.append(rid)
    raw = f"{stamp}-raw-confidence-final"
    sh("run_pipeline.py", *common, "--raw-confidence", "--fold", FINAL_FOLDS, "--run-id", raw)
    nocat = f"{stamp}-nocat-final"
    sh("run_pipeline.py", *common, "--hide-category", "--fold", FINAL_FOLDS, "--run-id", nocat)

    sh("evaluate.py", *db, "--run-id", full, "--folds", FINAL_FOLDS, "--compare", ",".join(ablation))
    for rid in (raw, nocat):
        sh("evaluate.py", *db, "--run-id", rid, "--folds", FINAL_FOLDS)
    sh("export.py", *db, "--run-id", full, "--out", str(paths.OUTPUTS / full))

    out = RESULTS / full
    out.mkdir(parents=True, exist_ok=True)
    src = paths.OUTPUTS / full
    for name in ("report.md", "run.json", "predictions.parquet"):
        shutil.copy2(src / name, out / name)
    for rid, name in ((raw, "report_raw_confidence.md"), (nocat, "report_step0_hidden_category.md")):
        shutil.copy2(paths.OUTPUTS / rid / "report.md", out / name)
    print(f"готово: {out.relative_to(paths.ROOT)}; цифры для results/README.md — из {out / 'report.md'}")


if __name__ == "__main__":
    main()
