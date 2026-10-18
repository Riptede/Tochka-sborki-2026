"""Возврат решений модератора в каскад (b5): синонимы — в словари, согласие с моделью — в пороги.

1. Синонимы. Решение по справочному атрибуту говорит, как написание из текста (value_raw) читается в каноне:
   accept подтверждает «написание → значение», fix даёт новое «написание → исправленное значение». Они пишутся
   в configs/review_synonyms.yaml в формате раздела synonyms из configs/normalize.yaml; нормализатор b2 берёт
   их вместе с ручными синонимами. Написание, по которому модераторы расходятся, не пишем.
2. Пороги. Доля accept среди решений по атрибуту при уверенности не ниже порога: если модераторы почти всегда
   согласны с моделью, порог атрибута можно опустить — список таких атрибутов печатается для решения команды.

    uv run python -m attrex.review.feedback
"""
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

from attrex.normalize.text import phrase_key
from attrex.registry import Registry

HEADER = "# Сгенерировано: python -m attrex.review.feedback из решений модераторов. Руками не править.\n"


def synonyms_from_decisions(decisions: pd.DataFrame, registry: Registry, min_votes: int = 1,
                            min_share: float = 0.8) -> dict[str, dict[str, str]]:
    """attr_id → {написание: канон} по решениям accept и fix у справочных атрибутов."""
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in decisions.itertuples():
        if row.decision not in ("accept", "fix") or not row.value or not row.value_raw:
            continue
        if row.attr_id not in registry.attrs or registry[row.attr_id].kind != "category":
            continue
        surface = phrase_key(str(row.value_raw))
        if surface and surface != phrase_key(str(row.value)):
            votes[row.attr_id, surface][row.value] += 1
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for (attr_id, surface), c in sorted(votes.items()):
        value, n = c.most_common(1)[0]
        if n >= min_votes and n / sum(c.values()) >= min_share:
            out[attr_id][surface] = value
    return dict(out)


def write_synonyms(synonyms: dict[str, dict[str, str]], path: Path) -> Path:
    path.write_text(HEADER + yaml.safe_dump({"synonyms": synonyms}, allow_unicode=True, sort_keys=True),
                    encoding="utf-8")
    return path


def agreement(decisions: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """По атрибуту: сколько решений и какая доля accept — в целом и среди значений с уверенностью ≥ медианы."""
    df = decisions.merge(predictions[["sku_id", "attr_id", "confidence"]], on=["sku_id", "attr_id"], how="left")
    rows = {}
    for attr_id, g in df.groupby("attr_id"):
        median = g.confidence.median()
        top = g[g.confidence >= median]
        rows[attr_id] = {"decisions": len(g), "accept": float((g.decision == "accept").mean()),
                         "median_conf": float(median), "accept_above_median": float((top.decision == "accept").mean())}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("decisions", ascending=False)


def main() -> None:
    import argparse

    import duckdb

    from attrex import paths
    from attrex.registry import load_registry
    from attrex.review.decisions import DecisionStore

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=paths.PROCESSED / "review" / "decisions.parquet")
    ap.add_argument("--out", type=Path, default=paths.CONFIGS / "review_synonyms.yaml")
    ap.add_argument("--db", type=Path, default=paths.DB)
    a = ap.parse_args()
    decisions = DecisionStore(a.store).load()
    if decisions.empty:
        raise SystemExit(f"решений нет: {a.store}")
    synonyms = synonyms_from_decisions(decisions, load_registry())
    write_synonyms(synonyms, a.out)
    print(f"{a.out}: синонимов {sum(map(len, synonyms.values()))} по {len(synonyms)} атрибутам")
    con = duckdb.connect(str(a.db), read_only=True)
    try:
        preds = con.execute("SELECT sku_id, attr_id, confidence FROM pred_predictions WHERE run_id IN "
                            "(SELECT unnest(?))", [sorted(decisions.run_id.unique())]).df()
    finally:
        con.close()
    table = agreement(decisions, preds)
    print(table.to_string(float_format=lambda v: f"{v:.2f}"))
    easy = table[(table.decisions >= 30) & (table.accept_above_median >= 0.95)]
    if len(easy):
        print("порог можно опустить (модераторы согласны с моделью ≥ 95%):", ", ".join(easy.index))


if __name__ == "__main__":
    main()
