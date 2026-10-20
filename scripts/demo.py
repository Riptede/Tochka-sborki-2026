"""Демо без данных организатора: весь путь решения на выборке реальных карточек из sample/raw/ за ~20 секунд.

    uv run python scripts/demo.py                                    # сборка БД → каскад → метрики → файл → очередь
    uv run python scripts/demo.py --title "Смартфон Samsung Galaxy A55 8/256GB синий" --description "Экран 6,6 дюйма"
    uv run python scripts/demo.py --db data/attrex.duckdb --title "…"  # своя карточка со словарями полной БД
    uv run python scripts/demo.py --title "…" --json                  # ответ в JSON для встраивания в сервис

Что получается в outputs/demo/:
    attrex.duckdb                 БД из выборки (таблицы raw_*, core_*, pred_*), как data/attrex.duckdb на полных данных
    demo/predictions.parquet|csv  файл предсказаний: строка на (карточка, обязательный атрибут), включая пустые
    demo/report.md                метрики на размеченной части выборки: P/R/F1 по атрибутам, срезы, вклад шагов
    demo/review_queue.csv         очередь ручной проверки: значения ниже порога, важные по трафику — первыми
    demo/review.html              интерфейс модератора по этой очереди: открыть в браузере

Режим одной карточки печатает по каждому обязательному атрибуту значение в каноне, уверенность, статус
(accepted — в каталог, review — на проверку, empty — в тексте не найдено) и цитату из текста, которая его подтверждает.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import duckdb

from attrex import paths
from attrex.contracts import BuildContext, Card, Prediction, Status, Step
from attrex.db.build import BuildConfig, build
from attrex.metrics.report import RunInfo, render
from attrex.metrics.scoring import comparison_same, coverage, evaluate, load_outcomes
from attrex.pipeline import build_pipeline, load_config
from attrex.registry import load_registry
from attrex.review.app import write_page
from attrex.review.queue import build_queue, write_queue
from attrex.runs import RunSpec, export, run, store

SAMPLE = paths.ROOT / "sample" / "raw"
OUT = paths.OUTPUTS / "demo"
RUN_ID = "demo"
EVAL_FOLDS = ("valid", "test", "holdout_category")   # словари строятся только по train выборки

_t0 = time.time()


def log(msg: str) -> None:
    # прогресс — в stderr: stdout режима --json остаётся чистым JSON
    print(f"[{time.time() - _t0:4.0f} с] {msg}", file=sys.stderr, flush=True)


def build_demo_db(db: Path) -> None:
    if not (SAMPLE / "skus.pkl").exists():
        raise SystemExit(f"нет выборки {SAMPLE}: она лежит в репозитории, проверьте git checkout")
    build(BuildConfig(raw_dir=SAMPLE, interim_dir=OUT / "interim", db_path=db, reparse=True, memory_limit="1GB",
                      threads=2), log=lambda m: None)
    log(f"БД из выборки: {db}")


def fmt_row(attr: str, value, unit, conf, status, step, evidence, gold: str | None = None) -> str:
    v = "—" if value is None else f"{value} {unit}" if unit else str(value)
    c = "" if conf is None else f"{conf:.2f}"
    ev = (evidence or "").replace("\n", " ")
    ev = f"«{ev[:38]}…»" if len(ev) > 38 else f"«{ev}»" if ev else ""
    g = "" if gold is None else f"{gold[:22]:<22} "
    return f"  {attr[:24]:<24} {v[:26]:<26} {c:>5}  {status:<9} {step:<12} {g}{ev}"


def header(gold: bool) -> str:
    g = f"{'эталон':<22} " if gold else ""
    return f"  {'атрибут':<24} {'значение':<26} {'увер.':>5}  {'статус':<9} {'шаг':<12} {g}цитата из текста"


def show_cards(con: duckdb.DuckDBPyConnection, n: int = 3) -> None:
    """Несколько карточек прогона: по одной с оффером из оценочных фолдов и карточки без оффера."""
    reg = load_registry()
    picks = con.execute(f"""
        WITH acc AS (SELECT sku_id, count(*) FILTER (WHERE status = 'accepted') AS n_acc, count(*) AS n_req
                     FROM pred_predictions WHERE run_id = '{RUN_ID}' GROUP BY 1)
        (SELECT c.sku_id FROM core_cards c JOIN acc USING (sku_id) JOIN core_splits s USING (sku_id)
         WHERE c.category_id IN (761, 987) AND s.fold IN ('valid', 'test', 'holdout_category')
         QUALIFY row_number() OVER (PARTITION BY c.category_id ORDER BY acc.n_acc DESC, c.sku_id) = 1
         ORDER BY c.category_id LIMIT {n - 1})
        UNION ALL
        (SELECT c.sku_id FROM core_cards c JOIN acc USING (sku_id) WHERE NOT c.has_offer
         ORDER BY acc.n_acc DESC, c.sku_id LIMIT 1)""").fetchall()
    for (sku,) in picks:
        title, cat, src = con.execute("""
            SELECT c.title, coalesce(r.name, '—'), coalesce(c.category_source, 'шаг 0 по заголовку')
            FROM core_cards c LEFT JOIN raw_categories r USING (category_id) WHERE sku_id = ?""", [sku]).fetchone()
        rows = con.execute("""
            SELECT p.attr_id, p.value, p.unit, p.confidence, p.status, p.step, p.evidence,
                   (SELECT string_agg(coalesce(g.value, 'заглушка'), ', ') FROM core_gold g
                    WHERE g.sku_id = p.sku_id AND g.attr_id = p.attr_id) AS gold,
                   (SELECT r.name FROM raw_categories r WHERE r.category_id = p.category_id) AS pred_cat
            FROM pred_predictions p WHERE p.run_id = ? AND p.sku_id = ? ORDER BY p.status, p.attr_id""",
                           [RUN_ID, sku]).fetchall()
        print(f"\n{title}\n  sku {sku}; категория: {rows[0][8] if rows else cat} (источник: {src})")
        print(header(gold=True))
        for a, v, u, c, st, step, ev, gold, _ in rows:
            print(fmt_row(reg[a].name_ru, v, u, c, st, step, ev, gold or ""))


def demo_run() -> None:
    db = OUT / "attrex.duckdb"
    OUT.mkdir(parents=True, exist_ok=True)
    build_demo_db(db)
    cfg = load_config()
    spec = RunSpec(RUN_ID, tuple(cfg["cascade"]), knowledge_folds=tuple(cfg["knowledge_folds"]))
    con = duckdb.connect(str(db), read_only=True)
    try:
        pipeline = build_pipeline(BuildContext(con, cfg, spec.knowledge_folds), spec.steps)
        result = run(pipeline, spec, con, OUT)
    finally:
        con.close()
    store(db, OUT / RUN_ID)
    log(f"каскад {' → '.join(pipeline.step_names)}: {result.n_cards} карточек, {result.n_rows} строк")

    con = duckdb.connect(str(db), read_only=True)
    try:
        metrics = evaluate(load_outcomes(con, EVAL_FOLDS, comparison_same(), run_id=RUN_ID))
        metrics.coverage = coverage(con, RUN_ID)
        info = RunInfo(RUN_ID, spec.steps, EVAL_FOLDS, result.n_cards,
                       note="Демо на выборке sample/raw: цифры иллюстративные, итоговые — в results/.")
        (OUT / RUN_ID / "report.md").write_text(render(info, metrics), encoding="utf-8")
        files = export(con, RUN_ID, OUT / RUN_ID)
        queue = write_queue(build_queue(con, RUN_ID), OUT / RUN_ID / "review_queue.csv")
        app = write_page(queue, RUN_ID, OUT / RUN_ID / "review.html", con)
        show_cards(con)
    finally:
        con.close()
    h = metrics.headline()
    print()
    log("метрики на размеченной части выборки (valid, test, holdout_category): "
        + ", ".join(f"{k} {v:.1%}" for k, v in h.items()))
    for p in [*files, OUT / RUN_ID / "report.md", queue, app]:
        print(f"       {p.relative_to(paths.ROOT)}")


def one_card(db: Path, title: str, description: str, as_json: bool = False) -> None:
    if not db.exists():
        if db != OUT / "attrex.duckdb":
            raise SystemExit(f"нет БД {db}: соберите её scripts/build_db.py или запустите demo.py без --db")
        build_demo_db(db)
    cfg = load_config()
    con = duckdb.connect(str(db), read_only=True)
    try:
        pipeline = build_pipeline(BuildContext(con, cfg, tuple(cfg["knowledge_folds"])))
        cats = dict(con.execute("SELECT category_id, name FROM raw_categories").fetchall())
    finally:
        con.close()
    card = Card(sku_id=0, title=title, description=description)
    resolved = pipeline.resolver.resolve(card) if pipeline.resolver else card
    preds = sorted(pipeline.run(card), key=lambda p: (str(p.status), p.attr_id))
    reg = load_registry()
    threshold = pipeline.aggregator.threshold
    category = cats.get(resolved.category_id)
    if as_json:
        print(json.dumps({"title": title, "category_id": resolved.category_id, "category": category,
                          "category_conf": resolved.category_conf,
                          "attributes": [{"attr_id": p.attr_id, "name": reg[p.attr_id].name_ru, "value": p.value,
                                          "unit": p.unit, "confidence": p.confidence,
                                          "threshold": threshold(p.attr_id), "status": str(p.status),
                                          "reason": reason(p, threshold(p.attr_id)), "step": str(p.step),
                                          "source": str(p.source), "evidence": p.evidence} for p in preds]},
                         ensure_ascii=False, indent=2))
        return
    conf = f"{resolved.category_conf:.2f}" if resolved.category_conf is not None else "—"
    print(f"\n{title}\n  категория: {category or 'не определена — только сквозные атрибуты'} (уверенность {conf})")
    print(header(gold=False))
    for p in preds:
        print(fmt_row(reg[p.attr_id].name_ru, p.value, p.unit, p.confidence, str(p.status), str(p.step), p.evidence))
    print("\n  accepted — в каталог; review — на ручную проверку; empty — в тексте не найдено, значение не выдумываем")
    for p in preds:
        if p.status is not Status.ACCEPTED:
            print(f"  {reg[p.attr_id].name_ru}: {reason(p, threshold(p.attr_id))}")


def reason(p: Prediction, threshold: float) -> str:
    """Почему значение принято или не принято — в словах модератора."""
    if p.status is Status.EMPTY:
        return "в тексте карточки значения нет"
    if p.status is Status.ACCEPTED:
        return f"уверенность {p.confidence:.2f} не ниже порога атрибута {threshold:.2f}, цитата подтверждена"
    if p.step is Step.MODEL_MATCH:
        return "значение взято у другой карточки той же модели — только на проверку"
    if p.confidence is not None and p.confidence < threshold:
        return f"уверенность {p.confidence:.3f} ниже порога атрибута {threshold:.3f} (точность ≥ 90% на valid)"
    return "цитата не подтвердила значение после нормализации — на проверку"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title", help="заголовок своей карточки: извлечь атрибуты только из неё")
    ap.add_argument("--description", default="", help="описание своей карточки")
    ap.add_argument("--db", type=Path, default=OUT / "attrex.duckdb", help="БД со словарями (по умолчанию из выборки)")
    ap.add_argument("--json", action="store_true", help="своя карточка: ответ в JSON — для встраивания в сервис")
    a = ap.parse_args()
    if a.title is not None:
        if not a.title.strip():
            ap.error("--title пустой: атрибуты извлекаются из заголовка и описания карточки")
        one_card(a.db, a.title, a.description, a.json)
    elif a.json:
        ap.error("--json — только для своей карточки: добавьте --title")
    else:
        demo_run()


if __name__ == "__main__":
    main()
