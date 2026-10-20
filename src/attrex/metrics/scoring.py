"""Метрики прогона (b4): P/R/F1 по атрибутам, macro по категориям, с весом по трафику, срезы, вклад шагов.

Пара (sku_id, attr_id) оценивается, только если у неё есть эталон и это не заглушка: атрибута может не быть в фиде,
хотя он есть в тексте, и такое предсказание нельзя считать ошибкой (EDA, раздел 4). Пары без эталона идут
только в покрытие.

    TP — accepted и совпало с эталоном (ValueNormalizer.same: числа с точностью округления, процессор по модели)
    FP — accepted и не совпало;   P = TP / (TP + FP);   R = TP / пар с эталоном
«До проверки» (P_any, R_any) — то же, но в счёт идут и значения со статусом review: качество каскада без порога.
R_text — полнота на парах, где значение эталона есть в тексте карточки (core_gold.in_text: title, description,
synonym): в тексте лежит только половина эталона, и полнота на всём эталоне занижает то, что достижимо без выдумки.
У многозначного атрибута значение верно, если совпало с любым значением эталона.
"""
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from attrex.normalize.dictionary import RefDictionary
from attrex.normalize.settings import load_settings
from attrex.normalize.values import ValueNormalizer
from attrex.registry import load_registry

SameFn = Callable[[str, str, str], bool]
SLICES = {"fold": "фолд", "seller_seg": "сегмент продавца", "seller_size": "размер продавца",
          "is_refurb": "восстановленные и уценённые", "category_source": "источник категории"}
MIN_GOLD_FOR_MACRO = 20     # категории с меньшим числом пар с эталоном в macro-F1 не входят: шум
IN_TEXT = ("title", "description", "synonym")   # core_gold.in_text: значение есть в тексте (partial и none — нет)
TOP_CATEGORIES = 20
HOLDOUT = "holdout_category"

_EVAL_SQL = """
    WITH g AS (
        SELECT sku_id, attr_id, list(value) FILTER (WHERE value IS NOT NULL) AS gold,
               bool_or(value IS NOT NULL AND in_text IN (SELECT unnest(?))) AS gold_in_text
        FROM core_gold GROUP BY 1, 2
    )
    SELECT p.sku_id, p.category_id, c.category_id AS true_category_id, p.attr_id, p.value, p.status, p.step,
           p.confidence, s.fold, c.seller_seg, coalesce(c.seller_size, 'нет') AS seller_size, c.is_refurb,
           coalesce(c.category_source, 'none') AS category_source, g.gold, coalesce(g.gold_in_text, false) AS in_text,
           coalesce(w.weight, 0) AS weight, r.name AS category_name
    FROM {predictions} p
    JOIN core_splits s USING (sku_id)
    JOIN core_cards c USING (sku_id)
    LEFT JOIN g USING (sku_id, attr_id)
    LEFT JOIN core_traffic_weights w ON w.category_id = p.category_id AND w.attr_id = p.attr_id
    LEFT JOIN raw_categories r ON r.category_id = p.category_id
    WHERE s.fold IN (SELECT unnest(?))
"""

# покрытие каталога: все пары (карточка, обязательный атрибут) прогона, а не только размеченные фолды
_COVERAGE_SQL = """
    WITH feed AS (
        SELECT DISTINCT sku_id, attr_id FROM core_gold WHERE value IS NOT NULL AND NOT coalesce(is_placeholder, false)
    )
    SELECT CASE WHEN c.has_offer THEN 'офферы' ELSE 'карточки без оффера' END AS part,
           count(DISTINCT p.sku_id) AS cards, count(*) AS pairs,
           avg(CASE WHEN f.sku_id IS NOT NULL THEN 1 ELSE 0 END) AS feed,
           avg(CASE WHEN p.status = 'accepted' THEN 1 ELSE 0 END) AS accepted,
           avg(CASE WHEN f.sku_id IS NOT NULL OR p.status = 'accepted' THEN 1 ELSE 0 END) AS feed_or_accepted,
           avg(CASE WHEN p.status <> 'empty' THEN 1 ELSE 0 END) AS any_value
    FROM pred_predictions p
    JOIN core_cards c USING (sku_id)
    LEFT JOIN feed f USING (sku_id, attr_id)
    WHERE p.run_id = ?
    GROUP BY ROLLUP (part)
"""


def comparison_same() -> SameFn:
    """Равенство значений для метрик — то же, что у нормализатора b2 (одна библиотека канона на весь проект)."""
    settings = load_settings()
    return ValueNormalizer(load_registry(), RefDictionary({}, {}), settings.scores, settings.key_functions).same


def load_outcomes(con: duckdb.DuckDBPyConnection, folds: Sequence[str], same: SameFn, run_id: str | None = None,
                  parquet: Path | None = None) -> pd.DataFrame:
    """Строка на предсказанную пару из фолдов folds: эталон, срезы, вес трафика и исход сравнения."""
    if parquet is not None:
        predictions, params = "read_parquet(?)", [str(parquet)]
    else:
        predictions, params = "(SELECT * FROM pred_predictions WHERE run_id = ?)", [run_id]
    df = con.execute(_EVAL_SQL.format(predictions=predictions), [list(IN_TEXT), *params, list(folds)]).df()
    df["gold"] = [list(g) if isinstance(g, (list, tuple, np.ndarray)) else [] for g in df.gold]   # NULL → []
    df["judged"] = df.gold.map(bool)
    df["in_text"] = df.in_text.astype(bool) & df.judged
    df["has_value"] = df.value.notna()
    df["accepted"] = df.status.eq("accepted")
    df["correct"] = [bool(j and isinstance(v, str) and any(same(a, v, x) for x in g))
                     for a, v, g, j in zip(df.attr_id, df.value, df.gold, df.judged)]
    return df


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r and not math.isnan(p + r) else 0.0


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def prf(df: pd.DataFrame) -> dict[str, float]:
    j = df[df.judged]
    tp = int((j.correct & j.accepted).sum())
    fp = int((j.accepted & ~j.correct).sum())
    tp_any = int(j.correct.sum())
    fp_any = int((j.has_value & ~j.correct).sum())
    t = j[j.in_text]
    p, r = _ratio(tp, tp + fp), _ratio(tp, len(j))
    p_any, r_any = _ratio(tp_any, tp_any + fp_any), _ratio(tp_any, len(j))
    return {"pairs": len(df), "gold": len(j), "accepted": float(df.accepted.mean()) if len(df) else float("nan"),
            "P": p, "R": r, "F1": _f1(p, r), "P_any": p_any, "R_any": r_any, "F1_any": _f1(p_any, r_any),
            "gold_text": len(t), "text_share": _ratio(len(t), len(j)),
            "R_text": _ratio(int((t.correct & t.accepted).sum()), len(t))}


def by(df: pd.DataFrame, col: str | list[str]) -> pd.DataFrame:
    rows = {k: prf(g) for k, g in df.groupby(col, dropna=False)}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("gold", ascending=False)


def macro_f1(df: pd.DataFrame, col: str = "category_id", min_gold: int = MIN_GOLD_FOR_MACRO) -> float:
    """Среднее F1 по категориям: длинный хвост весит столько же, сколько матрасы (EDA, раздел 3)."""
    t = by(df, col)
    t = t[t.gold >= min_gold]
    return float(t.F1.mean()) if len(t) else float("nan")


def traffic_f1(df: pd.DataFrame) -> float:
    """F1 пар (категория, атрибут) с весом по кликам core_traffic_weights: ошибки там, где ищут, стоят дороже."""
    t = by(df, ["category_id", "attr_id"])
    w = df.groupby(["category_id", "attr_id"]).weight.first().reindex(t.index).fillna(0)
    mask = (t.gold > 0) & (w > 0)
    return float((t.F1[mask] * w[mask]).sum() / w[mask].sum()) if mask.any() else float("nan")


def category_traffic(df: pd.DataFrame) -> pd.Series:
    """Доля кликов категории: сумма весов её обязательных атрибутов (core_traffic_weights делит клики поровну)."""
    return df.groupby(["category_id", "attr_id"]).weight.first().groupby(level=0).sum()


def category_table(df: pd.DataFrame, top: int = TOP_CATEGORIES) -> pd.DataFrame:
    """P/R/F1 по категориям: топ по трафику поимённо, остальные — одной строкой «хвост»; holdout_category — отдельно.

    Категории holdout_category целиком отложены как «новые»: их карточки в топ и хвост не входят.
    """
    if df.empty or "category_name" not in df:
        return pd.DataFrame()
    holdout = df.fold.eq(HOLDOUT)
    main = df[~holdout & df.category_id.notna()]
    traffic = category_traffic(main)
    judged = set(main.loc[main.judged, "category_id"])
    ranked = traffic[traffic.index.isin(judged) & (traffic > 0)].sort_values(ascending=False)
    top_ids = list(ranked.index[:top])
    names = df.groupby("category_id").category_name.first()
    rows = {}
    for cid in top_ids:
        rows[f"{names.get(cid) or '—'} ({int(cid)})"] = {"traffic": traffic[cid], **prf(main[main.category_id == cid])}
    tail = df[~holdout & ~df.category_id.isin(top_ids)]
    if len(tail):
        rows[f"хвост: {tail.category_id.nunique()} категорий"] = {
            "traffic": float(traffic.drop(top_ids, errors="ignore").sum()), **prf(tail)}
    if holdout.any():
        h = df[holdout]
        rows[f"{HOLDOUT}: {h.category_id.nunique()} категорий"] = {"traffic": float(category_traffic(h).sum()), **prf(h)}
    return pd.DataFrame.from_dict(rows, orient="index")


def coverage(con: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    """Покрытие каталога «до / после» по парам (карточка, обязательный атрибут) прогона.

    feed — значение есть в фиде (эталон без заглушек): «до»; accepted — значение каскада принято в каталог;
    feed_or_accepted — каталог после прогона, если принятые значения дополняют фид; any_value — каскад нашёл
    хоть что-то (accepted + review). Отдельно офферы, карточки без оффера (у них фида нет) и все карточки.
    """
    df = con.execute(_COVERAGE_SQL, [run_id]).df()
    if df.empty:
        return df
    df["part"] = df.part.fillna("все карточки")
    order = {"офферы": 0, "карточки без оффера": 1, "все карточки": 2}
    return df.sort_values("part", key=lambda s: s.map(order)).set_index("part")


def step_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Какой шаг дал итоговое значение: доля в принятых и точность на парах с эталоном."""
    acc = df[df.accepted]
    rows = {}
    for step, g in acc.groupby("step"):
        j = g[g.judged]
        rows[step] = {"rows": len(g), "share": len(g) / max(len(acc), 1), "gold": len(j),
                      "P": _ratio(int(j.correct.sum()), len(j))}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("rows", ascending=False)


def category_accuracy(df: pd.DataFrame) -> dict[str, float]:
    """Шаг 0: категория, с которой шёл прогон, против категории из БД — по карточкам (прогон с --hide-category)."""
    cards = df.drop_duplicates("sku_id")
    known = cards.true_category_id.notna()
    cards = cards[known]
    covered = cards.category_id.notna()
    hit = cards.category_id.eq(cards.true_category_id) & covered
    return {"cards": len(cards), "coverage": float(covered.mean()) if len(cards) else float("nan"),
            "accuracy": float(hit.mean()) if len(cards) else float("nan"),
            "accuracy_covered": float(hit[covered].mean()) if covered.any() else float("nan")}


@dataclass
class RunMetrics:
    overall: dict[str, float]
    macro_f1: float
    traffic_f1: float
    by_attr: pd.DataFrame
    slices: dict[str, pd.DataFrame] = field(default_factory=dict)
    steps: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: pd.Series = field(default_factory=pd.Series)
    category: dict[str, float] = field(default_factory=dict)
    categories: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)     # заполняет вызывающий: coverage(con, run_id)

    def headline(self) -> dict[str, float]:
        o = self.overall
        return {"P": o["P"], "R": o["R"], "F1": o["F1"], "R_text": o["R_text"], "macro_F1": self.macro_f1,
                "traffic_F1": self.traffic_f1, "accepted": o["accepted"], "F1_any": o["F1_any"]}


def evaluate(df: pd.DataFrame) -> RunMetrics:
    return RunMetrics(
        overall=prf(df), macro_f1=macro_f1(df), traffic_f1=traffic_f1(df), by_attr=by(df, "attr_id"),
        slices={col: by(df, col) for col in SLICES if df[col].nunique(dropna=False) > 1},
        steps=step_contribution(df), status=df.status.value_counts(normalize=True),
        category=category_accuracy(df), categories=category_table(df),
    )
