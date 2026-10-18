"""Интерфейс модератора (b5): HTML-страница по очереди review_queue.csv — открывается файлом, без сервера.

    uv run python -m attrex.review.app --run-id <run_id>                 # outputs/<run_id>/review.html
    uv run python -m attrex.review.app --run-id demo --out outputs/demo --db outputs/demo/attrex.duckdb   # демо

На карточке: заголовок и описание с подсвеченной цитатой, значение каскада, уверенность и порог атрибута.
Решения — по docs/review_protocol.md: accept, fix (верное значение: из справочника core_ref_values с подсказками
при вводе или число в единице атрибута), absent. Черновик решений хранится в браузере. «Скачать решения»
выгружает CSV в формате очереди (те же колонки QUEUE_COLUMNS, заполнены decision и fixed_value): его без правок
принимает python -m attrex.review.decisions. Описания и справочник берутся из БД (только чтение); без БД
страница строится по одной очереди — с заголовком и цитатой, но без описания и подсказок.
"""
import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from attrex.paths import CONFIGS
from attrex.registry import Registry, load_registry
from attrex.review.queue import QUEUE_COLUMNS

TEMPLATE = Path(__file__).with_name("app.html")
PLACEHOLDER = "/*__DATA__*/"
MAX_OPTIONS = 2000      # подсказок на атрибут: справочник бренда — тысячи значений, в странице — самые частые


def read_queue(path: Path) -> pd.DataFrame:
    q = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in QUEUE_COLUMNS if c not in q.columns]
    if missing:
        raise ValueError(f"{path}: это не очередь проверки, нет колонок {missing}")
    return q[QUEUE_COLUMNS]


def thresholds(registry: Registry) -> dict[str, float]:
    """Порог «в каталог» по атрибуту — как у агрегатора: калибровка, поверх неё ручные пороги конфига."""
    pipeline = yaml.safe_load((CONFIGS / "pipeline.yaml").read_text(encoding="utf-8"))["confidence"]
    out: dict[str, float] = {}
    if pipeline.get("calibration") and (CONFIGS / pipeline["calibration"]).exists():
        out.update(yaml.safe_load((CONFIGS / pipeline["calibration"]).read_text(encoding="utf-8")).get("thresholds") or {})
    out.update(pipeline.get("thresholds") or {})
    return {a: float(out.get(a, pipeline["accept_threshold"])) for a in registry.attrs}


def _from_db(con: duckdb.DuckDBPyConnection, q: pd.DataFrame) -> tuple[dict, dict]:
    skus = sorted({int(s) for s in q.sku_id})
    cards = con.execute("""
        SELECT c.sku_id, coalesce(c.description, ''), r.name
        FROM core_cards c LEFT JOIN raw_categories r USING (category_id)
        WHERE c.sku_id IN (SELECT unnest(?))""", [skus]).fetchall()
    options: dict[str, list[str]] = {}
    for attr_id, value in con.execute("""
            SELECT attr_id, value FROM core_ref_values WHERE value IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY attr_id ORDER BY n_offers DESC, value) <= ?
            ORDER BY attr_id, n_offers DESC, value""", [MAX_OPTIONS]).fetchall():
        options.setdefault(attr_id, []).append(value)
    return {str(s): {"description": d, "category": n} for s, d, n in cards}, options


def page_data(queue_path: Path, run_id: str, con: duckdb.DuckDBPyConnection | None = None) -> dict:
    """Всё, что нужно странице: строки очереди как есть (строками), тексты карточек, атрибуты и справочник."""
    registry = load_registry()
    q = read_queue(queue_path)
    cards, options = _from_db(con, q) if con is not None else ({}, {})
    th = thresholds(registry)
    attrs = {a: {"name": s.name_ru, "kind": s.kind, "unit": s.unit, "threshold": th[a]} for a, s in registry.attrs.items()}
    options = {a: v for a, v in options.items() if a in attrs and attrs[a]["kind"] == "category"}
    stem = f"{queue_path.stem}_{run_id}"
    return {
        "run_id": run_id, "queue_file": queue_path.name, "columns": QUEUE_COLUMNS,
        "rows": q.to_dict(orient="records"), "cards": cards, "has_descriptions": con is not None,
        "attrs": attrs, "options": options,
        "download_name": f"{stem}_decisions.csv",
        "load_command": f"uv run python -m attrex.review.decisions --run-id {run_id} --file <файл> --reviewer <имя>",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def render_page(data: dict) -> str:
    # «<» в JSON — <: текст карточки с «</script>» не закроет блок данных
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, payload)


def write_page(queue_path: Path, run_id: str, out: Path, con: duckdb.DuckDBPyConnection | None = None) -> Path:
    out.write_text(render_page(page_data(queue_path, run_id, con)), encoding="utf-8")
    return out


def main() -> None:
    import argparse

    from attrex import paths

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", type=Path, default=paths.OUTPUTS, help="папка прогонов: очередь — <out>/<run_id>/")
    ap.add_argument("--queue", type=Path, help="очередь, если не <out>/<run_id>/review_queue.csv")
    ap.add_argument("--db", type=Path, default=paths.DB, help="описания и справочник (только чтение)")
    a = ap.parse_args()
    queue = a.queue or a.out / a.run_id / "review_queue.csv"
    if not queue.exists():
        raise SystemExit(f"нет очереди {queue}: соберите её python -m attrex.review.queue --run-id {a.run_id}")
    con = duckdb.connect(str(a.db), read_only=True) if a.db.exists() else None
    try:
        path = write_page(queue, a.run_id, queue.with_name("review.html"), con)
    finally:
        if con is not None:
            con.close()
    print(f"{path}: откройте в браузере{'' if con else ' (без БД: без описаний и подсказок справочника)'}")


if __name__ == "__main__":
    main()
