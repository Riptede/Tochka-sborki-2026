"""Прогон каскада (b4): карточки из data/attrex.duckdb → outputs/<run_id>/ → pred_predictions, pred_runs.

    uv run python scripts/run_pipeline.py --steps rules --fold valid           # базлайн «только правила» на valid
    uv run python scripts/run_pipeline.py --steps rules,parse --fold valid     # + парсер — строка ablation
    uv run python scripts/run_pipeline.py --workers 3                          # все карточки — для файла сдачи
    uv run python scripts/run_pipeline.py --load-only --run-id 20261018-0300-rules-valid   # загрузить готовый прогон

На ВМ хакатона: перед запуском `free -h`, долгий прогон — в tmux (CLAUDE.md, «Где работаем»).
"""
import argparse
import sys
import time
from pathlib import Path

import duckdb

from attrex import paths
from attrex.contracts import BuildContext
from attrex.db import connect
from attrex.pipeline import build_pipeline, load_config
from attrex.runs import RunSpec, default_run_id, parse_steps, run, store


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip()) if value else ()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", help="по умолчанию YYYYMMDD-HHMM-<шаги>[-<фолды>]")
    ap.add_argument("--steps", help="шаги каскада через запятую (configs/pipeline.yaml, cascade); по умолчанию все")
    ap.add_argument("--fold", help="только карточки этих фолдов core_splits через запятую: valid, test, …")
    ap.add_argument("--category", type=int, help="только карточки этой категории")
    ap.add_argument("--limit", type=int, help="первые N карточек — для быстрой проверки")
    ap.add_argument("--workers", type=int, default=1, help="процессов для прогона")
    ap.add_argument("--hide-category", action="store_true",
                    help="скрыть категорию из БД: офферы идут как карточки без оффера (проверка шага 0)")
    ap.add_argument("--raw-confidence", action="store_true",
                    help="без калибровки: сырая уверенность и порог accept_threshold (строка ablation «до калибровки»)")
    ap.add_argument("--knowledge-folds", help="из эталона каких фолдов строить словари (по умолчанию из конфига)")
    ap.add_argument("--db", type=Path, default=paths.DB)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS)
    ap.add_argument("--no-store", action="store_true", help="не писать в БД, только outputs/<run_id>/")
    ap.add_argument("--load-only", action="store_true", help="загрузить готовый outputs/<run_id>/ в БД")
    a = ap.parse_args()

    cfg = load_config()
    if a.load_only:
        if not a.run_id:
            ap.error("--load-only требует --run-id")
        store(a.db, a.out / a.run_id)
        print(f"загружено: {a.run_id}")
        return

    if a.raw_confidence:
        cfg = {**cfg, "confidence": {**cfg["confidence"], "calibration": None}}
    steps = parse_steps(a.steps, list(cfg["cascade"]))
    if "llm" in steps and cfg["llm"].get("enabled") and a.workers > 1:
        ap.error("LLM-прогон — с --workers 1: бюджет llm.budget_rub считается на процесс")
    folds = _csv(a.fold)
    knowledge = _csv(a.knowledge_folds) or tuple(cfg["knowledge_folds"])
    if set(folds) & set(knowledge):
        print(f"внимание: фолды {sorted(set(folds) & set(knowledge))} — и оценка, и знания; метрики будут завышены",
              file=sys.stderr)
    label = "-".join([*steps, *folds, *(["nocat"] if a.hide_category else [])])
    spec = RunSpec(run_id=a.run_id or default_run_id(label), steps=steps, knowledge_folds=knowledge, folds=folds,
                   category_id=a.category, limit=a.limit, workers=a.workers, hide_category=a.hide_category,
                   extra={"calibration": cfg["confidence"].get("calibration")})

    t0 = time.time()
    con = connect(a.db)
    try:
        pipeline = build_pipeline(BuildContext(con, cfg, knowledge), steps)
        print(f"[{time.time() - t0:5.0f} с] пайплайн: {' → '.join(pipeline.step_names)}")
        result = run(pipeline, spec, con, a.out)
    finally:
        con.close()
    print(f"[{time.time() - t0:5.0f} с] {spec.run_id}: {result.n_cards:,} карточек, {result.n_rows:,} строк → "
          f"{result.parquet}")
    if a.no_store:
        return
    try:
        store(a.db, a.out / spec.run_id)
        print(f"[{time.time() - t0:5.0f} с] записано в pred_predictions, pred_runs")
    except duckdb.IOException as e:   # БД держит другой процесс: прогон не теряем
        print(f"БД занята ({e}); загрузить позже: --load-only --run-id {spec.run_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
