"""Прогон пайплайна (b4): карточки из БД → предсказания → outputs/<run_id>/ и pred_predictions, pred_runs.

Пока идёт прогон, БД открыта только на чтение: карточки читаются потоком, предсказания пишутся в parquet.
Писатель открывается в самом конце и на секунды (DuckDB пускает одного писателя, CLAUDE.md). Если БД занята,
предсказания остаются в parquet, и их можно загрузить позже: `run_pipeline.py --load-only --run-id ...`.
"""
import json
import os
import subprocess
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from attrex.contracts import PREDICTION_COLUMNS, Card, CategorySource, Usage
from attrex.paths import ROOT
from attrex.pipeline import Pipeline

BATCH_CARDS = 2_000

PREDICTION_SCHEMA = pa.schema([
    ("sku_id", pa.int64()), ("category_id", pa.int32()), ("category_conf", pa.float64()), ("attr_id", pa.string()),
    ("value", pa.string()), ("unit", pa.string()), ("confidence", pa.float64()), ("status", pa.string()),
    ("step", pa.string()), ("source", pa.string()), ("value_raw", pa.string()), ("evidence", pa.string()),
])
assert PREDICTION_SCHEMA.names == PREDICTION_COLUMNS


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    steps: tuple[str, ...]
    knowledge_folds: tuple[str, ...] = ("train",)
    folds: tuple[str, ...] = ()            # только карточки этих фолдов core_splits (оценка); пусто — все карточки
    categories: tuple[int, ...] = ()      # только карточки этих категорий; пусто — все
    limit: int | None = None
    workers: int = 1
    hide_category: bool = False            # скрыть категорию из БД: офферы как карточки без оффера (проверка шага 0)
    extra: dict = field(default_factory=dict)   # что ещё записать в pred_runs.config


def default_run_id(label: str, now: datetime | None = None) -> str:
    """run_id = YYYYMMDD-HHMM-<коротко> (CLAUDE.md, Нейминг)."""
    return f"{(now or datetime.now()):%Y%m%d-%H%M}-{label}"


def iter_cards(con: duckdb.DuckDBPyConnection, spec: RunSpec, batch: int = BATCH_CARDS) -> Iterator[list[Card]]:
    where, params = [], []
    join = ""
    if spec.folds:
        join = "JOIN core_splits s USING (sku_id)"
        where.append("s.fold IN (SELECT unnest(?))")
        params.append(list(spec.folds))
    if spec.categories:
        where.append("c.category_id IN (SELECT unnest(?))")
        params.append(list(spec.categories))
    sql = f"""
        SELECT c.sku_id, c.title, coalesce(c.description, ''), c.category_id, c.category_conf, c.category_source
        FROM core_cards c {join} {"WHERE " + " AND ".join(where) if where else ""}
        ORDER BY c.sku_id {f"LIMIT {int(spec.limit)}" if spec.limit else ""}"""
    reader = con.execute(sql, params).to_arrow_reader(batch)
    for rb in reader:
        cols = rb.to_pydict()
        if spec.hide_category:
            yield [Card(sku_id=s, title=t or "", description=d) for s, t, d, *_ in zip(*cols.values())]
            continue
        yield [Card(sku_id=s, title=t or "", description=d, category_id=c, category_conf=cc,
                    category_source=CategorySource(src) if src else None)
               for s, t, d, c, cc, src in zip(*cols.values())]


_worker_pipeline: Pipeline | None = None


def _init_worker(pipeline: Pipeline) -> None:
    global _worker_pipeline
    _worker_pipeline = pipeline


def _run_batch(cards: list[Card]) -> tuple[list[dict], Usage]:
    rows = [p.to_row() for card in cards for p in _worker_pipeline.run(card)]
    return rows, _worker_pipeline.drain_usage()


def predict(pipeline: Pipeline, batches: Iterable[list[Card]], workers: int = 1) -> Iterator[tuple[list[dict], Usage]]:
    """Предсказания по пачкам карточек в исходном порядке; workers > 1 — параллельно в процессах.

    В очереди не больше 2·workers пачек: Executor.map забрал бы все карточки в память сразу.
    """
    if workers <= 1:
        _init_worker(pipeline)
        yield from map(_run_batch, batches)
        return
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(pipeline,)) as pool:
        pending: deque[Future] = deque()
        for cards in batches:
            pending.append(pool.submit(_run_batch, cards))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


@dataclass
class RunResult:
    run_id: str
    parquet: Path
    n_cards: int = 0
    n_rows: int = 0
    usage: Usage = field(default_factory=Usage)
    started_at: datetime = field(default_factory=datetime.now)
    stored: bool = False


def run(pipeline: Pipeline, spec: RunSpec, con: duckdb.DuckDBPyConnection, out_dir: Path) -> RunResult:
    """Прогон по карточкам спецификации; результат — outputs/<run_id>/predictions.parquet."""
    out = out_dir / spec.run_id
    out.mkdir(parents=True, exist_ok=True)
    result = RunResult(spec.run_id, out / "predictions.parquet")
    n_cards = 0

    def counted() -> Iterator[list[Card]]:
        nonlocal n_cards
        for cards in iter_cards(con, spec):
            n_cards += len(cards)
            yield cards

    with pq.ParquetWriter(result.parquet, PREDICTION_SCHEMA) as writer:
        for rows, usage in predict(pipeline, counted(), spec.workers):
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, PREDICTION_SCHEMA))
            result.n_rows += len(rows)
            result.usage = result.usage + usage
    result.n_cards = n_cards
    (out / "run.json").write_text(json.dumps(_run_config(spec, result), ensure_ascii=False, indent=2, default=str),
                                  encoding="utf-8")
    return result


def _run_config(spec: RunSpec, result: RunResult) -> dict:
    return {**asdict(spec), "n_cards": result.n_cards, "n_rows": result.n_rows, "usage": asdict(result.usage),
            "started_at": result.started_at.isoformat(timespec="seconds"), "git_commit": git_commit()}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def store(db_path: Path, run_dir: Path) -> None:
    """Загрузить прогон из outputs/<run_id>/ в pred_predictions и pred_runs (заменяет прогон с тем же run_id)."""
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    usage = meta["usage"]
    con = duckdb.connect(str(db_path))
    try:
        con.execute("BEGIN")
        con.execute("DELETE FROM pred_predictions WHERE run_id = ?", [meta["run_id"]])
        con.execute("DELETE FROM pred_runs WHERE run_id = ?", [meta["run_id"]])
        con.execute(f"INSERT INTO pred_predictions SELECT ? AS run_id, {', '.join(PREDICTION_COLUMNS)} "
                    "FROM read_parquet(?)", [meta["run_id"], str(run_dir / "predictions.parquet")])
        con.execute("INSERT INTO pred_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [meta["run_id"], meta["started_at"], meta["git_commit"], json.dumps(meta, ensure_ascii=False),
                     usage["llm_calls"], usage["llm_tokens"], usage["llm_cost_rub"], usage["embed_tokens"],
                     usage["embed_cost_rub"]])
        con.execute("COMMIT")
    finally:
        con.close()


def export(con: duckdb.DuckDBPyConnection, run_id: str, out_dir: Path) -> list[Path]:
    """Файл сдачи: одна строка на пару (sku_id, attr_id), колонки PREDICTION_COLUMNS, parquet и csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = con.execute("SELECT count(*) FROM pred_predictions WHERE run_id = ?", [run_id]).fetchone()[0]
    if not n:
        raise ValueError(f"в pred_predictions нет прогона {run_id}")
    # COPY не принимает параметры запроса: run_id подставляем экранированным литералом
    select = (f"SELECT {', '.join(PREDICTION_COLUMNS)} FROM pred_predictions "
              f"WHERE run_id = {_sql_str(run_id)} ORDER BY sku_id, attr_id")
    paths = [out_dir / "predictions.parquet", out_dir / "predictions.csv"]
    for path, fmt in zip(paths, ("FORMAT parquet, COMPRESSION zstd", "FORMAT csv, HEADER")):
        tmp = path.with_name(path.name + ".tmp")
        con.execute(f"COPY ({select}) TO {_sql_str(str(tmp).replace(chr(92), '/'))} ({fmt})")
        os.replace(tmp, path)
    return paths


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def parse_steps(value: str | None, cascade: Sequence[str]) -> tuple[str, ...]:
    """«rules,parse» → ('rules', 'parse'); пусто — весь каскад."""
    if not value:
        return tuple(cascade)
    return tuple(s.strip() for s in value.split(",") if s.strip())
