"""Пути проекта. Все модули берут пути отсюда, а не собирают их сами."""
import os
from pathlib import Path


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("не найден корень репозитория (pyproject.toml)")


def load_env(path: Path) -> list[str]:
    """Переменные из .env (шаблон — .env.example): KEY=VALUE, # — комментарий. Уже заданные в окружении
    и пустые значения не трогаем. Возвращает имена загруженных переменных."""
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (s.strip() for s in line.removeprefix("export ").split("=", 1))
        value = value.strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


ROOT = _find_root()
load_env(ROOT / ".env")
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
