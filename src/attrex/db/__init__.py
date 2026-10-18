"""b1. Сборка БД data/attrex.duckdb из файлов организатора: raw_ → core_ таблицы (schema.sql)."""
from pathlib import Path

import duckdb

NO_DB_HINT = ("Соберите её из файлов организатора в data/raw/: uv run python scripts/build_db.py\n"
              "Без данных организатора весь путь показывает демо: uv run python scripts/demo.py")
RAW_FILES = ("skus.pkl", "sku_desc.parquet", "query_clicks.parquet")


class DbError(SystemExit):
    """Понятная ошибка вместо трассировки: скрипт печатает сообщение и завершается с кодом 1."""

    def __init__(self, message: str):
        super().__init__(f"ошибка: {message}")


def connect(path: Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Открыть data/attrex.duckdb. Нет файла или его держит писатель — DbError с подсказкой, что сделать."""
    if not Path(path).exists():
        raise DbError(f"нет БД {path}.\n{NO_DB_HINT}")
    try:
        return duckdb.connect(str(path), read_only=read_only)
    except duckdb.IOException as e:
        raise DbError(f"БД {path} занята другим процессом: писать в неё может только один build_db.py или "
                      f"run_pipeline.py (CLAUDE.md). Дождитесь его окончания.\n{e}") from e


def require_raw(raw_dir: Path, need_feed: bool = True) -> None:
    """Файлы организатора на месте; skus.pkl не нужен, если его разбор уже есть в кэше."""
    names = RAW_FILES if need_feed else RAW_FILES[1:]
    missing = [n for n in names if not (Path(raw_dir) / n).exists()]
    if missing:
        raise DbError(f"в {raw_dir} нет файлов организатора: {', '.join(missing)}.\n"
                      "Положите их туда (README, «Полный запуск»). Без данных: uv run python scripts/demo.py")
