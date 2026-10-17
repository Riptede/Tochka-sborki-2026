"""Пути проекта. Все модули берут пути отсюда, а не собирают их сами."""
import os
from pathlib import Path


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("не найден корень репозитория (pyproject.toml)")


ROOT = _find_root()
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = Path(os.environ.get("ATTREX_RAW", DATA / "raw"))      # файлы организатора, только чтение
INTERIM = DATA / "interim"                                    # кэш разбора сырых файлов (EDA, b1)
PROCESSED = DATA / "processed"                                # индексы, эмбеддинги, модели
DB = Path(os.environ.get("ATTREX_DB", DATA / "attrex.duckdb"))
OUTPUTS = ROOT / "outputs"                                    # predictions и отчёты для сдачи

RAW_SKUS = RAW / "skus.pkl"
RAW_DESC = RAW / "sku_desc.parquet"
RAW_CLICKS = RAW / "query_clicks.parquet"
