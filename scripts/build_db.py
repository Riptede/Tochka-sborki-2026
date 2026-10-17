"""Сборка data/attrex.duckdb из файлов организатора (блок b1).

    uv run python scripts/build_db.py                 # кэш разбора фида берётся из data/interim/, если есть
    uv run python scripts/build_db.py --reparse       # заново разобрать skus.pkl
    uv run python scripts/build_db.py --fast-pickle   # загрузить skus.pkl целиком (≈10 ГБ RAM; не на ВМ)

На ВМ хакатона запускать в tmux и при свободной памяти: см. CLAUDE.md, раздел «Где работаем».
"""
import argparse
from pathlib import Path

from attrex import paths
from attrex.db.build import BuildConfig, build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=paths.RAW, help="папка с файлами организатора")
    ap.add_argument("--interim", type=Path, default=paths.INTERIM, help="кэш разбора skus.pkl")
    ap.add_argument("--db", type=Path, default=paths.DB, help="куда записать БД")
    ap.add_argument("--reparse", action="store_true", help="разобрать skus.pkl заново, даже если кэш есть")
    ap.add_argument("--fast-pickle", action="store_true", help="загрузить skus.pkl целиком (нужно ≈10 ГБ RAM)")
    ap.add_argument("--memory-limit", default="3GB", help="лимит памяти DuckDB")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    counts = build(BuildConfig(raw_dir=a.raw, interim_dir=a.interim, db_path=a.db, reparse=a.reparse,
                               fast_pickle=a.fast_pickle, memory_limit=a.memory_limit, threads=a.threads))
    width = max(map(len, counts))
    for t, n in counts.items():
        print(f"  {t:<{width}}  {n:>12,}")


if __name__ == "__main__":
    main()
