"""Отчёт прогона в markdown: outputs/<run_id>/report.md. Цифры для README и презентации берём только отсюда."""
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import pandas as pd

from attrex.metrics.scoring import SLICES, RunMetrics

PCT = {"accepted", "P", "R", "F1", "P_any", "R_any", "F1_any", "macro_F1", "traffic_F1", "share", "coverage",
       "accuracy", "accuracy_covered"}


@dataclass
class RunInfo:
    run_id: str
    steps: Sequence[str]
    folds: Sequence[str]
    n_cards: int | None = None
    cost_rub: float = 0.0
    note: str = ""


def _fmt(col: str, v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, bool) or not isinstance(v, Real):
        return str(v)
    if col in PCT:
        return f"{v:.1%}"
    if isinstance(v, Integral) or float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}"


def md_table(df: pd.DataFrame, columns: Sequence[str], index_name: str = "") -> str:
    cols = [c for c in columns if c in df.columns]
    lines = ["| " + " | ".join([index_name, *cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx), *(_fmt(c, row[c]) for c in cols)]) + " |")
    return "\n".join(lines)


def ablation_table(runs: Sequence[tuple[RunInfo, RunMetrics]]) -> str:
    df = pd.DataFrame([{"steps": " → ".join(i.steps), **m.headline(), "cost_rub": i.cost_rub} for i, m in runs],
                      index=[i.run_id for i, _ in runs])
    return md_table(df, ["steps", "P", "R", "F1", "macro_F1", "traffic_F1", "accepted", "F1_any", "cost_rub"], "run_id")


def render(info: RunInfo, m: RunMetrics, ablation: Sequence[tuple[RunInfo, RunMetrics]] = ()) -> str:
    out = [
        f"# Прогон {info.run_id}",
        "",
        f"Шаги: {' → '.join(info.steps) or '—'}. Фолды оценки: {', '.join(info.folds)}."
        + (f" Карточек в прогоне: {info.n_cards:,}." if info.n_cards else "")
        + (f" Стоимость: {info.cost_rub:,.2f} ₽." if info.cost_rub else ""),
        "",
        ("Оцениваются пары (карточка, атрибут) с эталоном; P и R — по значениям со статусом accepted, "
         "«до проверки» (_any) — по всем непустым. Определения — `src/attrex/metrics/scoring.py`."),
    ]
    if info.note:
        out += ["", info.note]
    out += ["", "## Итог", "", md_table(pd.DataFrame([m.headline()], index=["все"]),
                                        ["P", "R", "F1", "macro_F1", "traffic_F1", "accepted", "F1_any"])]
    if ablation:
        out += ["", "## Ablation: вклад шагов", "", ablation_table(ablation)]
    cat = m.category
    if cat and cat.get("cards") and (cat["accuracy"] < 1 or cat["coverage"] < 1):   # категория шла не из БД
        out += ["", "## Шаг 0: категория", "",
                md_table(pd.DataFrame([cat], index=["карточки"]), ["cards", "coverage", "accuracy", "accuracy_covered"])]
    out += ["", "## По атрибутам", "",
            md_table(m.by_attr, ["gold", "accepted", "P", "R", "F1", "P_any", "R_any"], "attr_id")]
    for col, title in SLICES.items():
        if col in m.slices:
            out += ["", f"## Срез: {title}", "", md_table(m.slices[col], ["gold", "accepted", "P", "R", "F1"], col)]
    if len(m.steps):
        out += ["", "## Какой шаг дал принятое значение", "", md_table(m.steps, ["rows", "share", "gold", "P"], "step")]
    if len(m.status):
        out += ["", "## Статусы", "", md_table(m.status.to_frame("share"), ["share"], "status")]
    return "\n".join(out) + "\n"
