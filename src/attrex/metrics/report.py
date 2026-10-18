"""Отчёт прогона в markdown: outputs/<run_id>/report.md. Цифры для README и презентации берём только отсюда."""
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real

import pandas as pd

from attrex.metrics.scoring import SLICES, RunMetrics

PCT = {"accepted", "P", "R", "F1", "P_any", "R_any", "F1_any", "macro_F1", "traffic_F1", "share", "coverage",
       "accuracy", "accuracy_covered", "R_text", "text_share", "traffic", "feed", "feed_or_accepted", "any_value"}


@dataclass
class RunInfo:
    run_id: str
    steps: Sequence[str]
    folds: Sequence[str]
    n_cards: int | None = None
    cost_rub: float = 0.0
    note: str = ""
    usage: dict = field(default_factory=dict)     # run.json → usage: расход и счётчики LLM-шага

    @property
    def rub_per_1k(self) -> float:
        """Стоимость на 1 000 карточек прогона."""
        return self.cost_rub / self.n_cards * 1000 if self.n_cards else float("nan")


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


HEADLINE = ["P", "R", "F1", "R_text", "macro_F1", "traffic_F1", "accepted", "F1_any"]


def ablation_table(runs: Sequence[tuple[RunInfo, RunMetrics]]) -> str:
    df = pd.DataFrame([{"steps": " → ".join(i.steps), **m.headline(), "cost_rub": i.cost_rub,
                        "rub_per_1k": i.rub_per_1k} for i, m in runs], index=[i.run_id for i, _ in runs])
    return md_table(df, ["steps", *HEADLINE, "cost_rub", "rub_per_1k"], "run_id")


def llm_table(info: RunInfo) -> pd.DataFrame:
    """Расход и качество LLM-шага по счётчикам Usage (llm/client.py, llm/extractor.py)."""
    u = info.usage
    calls, values, cards = u.get("llm_calls", 0), u.get("llm_values", 0), u.get("llm_cards", 0)
    cost = u.get("llm_cost_rub", 0.0)

    def share(a, b):
        return f"{a:,} ({a / b:.1%})" if b else f"{a:,}"

    rows = [
        ("карточек ушло в LLM", share(cards, info.n_cards or 0)),
        ("запросов к API", f"{calls:,}"),
        ("токенов", f"{u.get('llm_tokens', 0):,}"),
        ("стоимость LLM, ₽", f"{cost:,.2f}"),
        ("₽ на 1 000 карточек прогона", f"{cost / info.n_cards * 1000:,.2f}" if info.n_cards else "—"),
        ("₽ на 1 000 карточек, ушедших в LLM", f"{cost / cards * 1000:,.2f}" if cards else "—"),
        ("сбоев API после повторов", f"{u.get('api_failures', 0):,}"),
        ("запросов пропущено (бюджет, сбои подряд)", f"{u.get('llm_skipped', 0):,}"),
        ("ответов не по JSON-схеме", share(u.get("llm_invalid", 0), calls)),
        ("непустых значений в ответах", f"{values:,}"),
        ("из них отброшено проверкой цитаты", share(u.get("llm_ungrounded", 0), values)),
    ]
    return pd.DataFrame({"value": [v for _, v in rows]}, index=[k for k, _ in rows])


def render(info: RunInfo, m: RunMetrics, ablation: Sequence[tuple[RunInfo, RunMetrics]] = ()) -> str:
    cost = ""
    if info.cost_rub:
        cost = f" Стоимость: {info.cost_rub:,.2f} ₽" + (f" ({info.rub_per_1k:,.2f} ₽ на 1 000 карточек)."
                                                        if info.n_cards else ".")
    o = m.overall
    out = [
        f"# Прогон {info.run_id}",
        "",
        f"Шаги: {' → '.join(info.steps) or '—'}. Фолды оценки: {', '.join(info.folds)}."
        + (f" Карточек в прогоне: {info.n_cards:,}." if info.n_cards else "") + cost,
        "",
        ("Оцениваются пары (карточка, атрибут) с эталоном; P и R — по значениям со статусом accepted, "
         "«до проверки» (_any) — по всем непустым. R_text — полнота на парах, где значение эталона есть в тексте "
         "карточки (core_gold.in_text: title, description, synonym)"
         + (f": таких {o['gold_text']:,} из {o['gold']:,} ({o['text_share']:.1%})" if o["gold"] else "")
         + ". Определения — `src/attrex/metrics/scoring.py`."),
    ]
    if info.note:
        out += ["", info.note]
    out += ["", "## Итог", "", md_table(pd.DataFrame([m.headline()], index=["все"]), HEADLINE)]
    if ablation:
        out += ["", "## Ablation: вклад шагов", "", ablation_table(ablation)]
    if info.usage.get("llm_cards"):
        out += ["", "## LLM-шаг: расход и проверка ответов", "", md_table(llm_table(info), ["value"], "")]
    if len(m.coverage):
        out += ["", "## Покрытие каталога: до и после", "",
                ("Пары (карточка, обязательный атрибут) всех карточек прогона. feed — значение есть в фиде "
                 "(«до»), accepted — принято каскадом, feed_or_accepted — фид, дополненный принятыми значениями "
                 "(«после»), any_value — каскад нашёл значение (accepted или review)."), "",
                md_table(m.coverage, ["cards", "pairs", "feed", "accepted", "feed_or_accepted", "any_value"], "")]
    cat = m.category
    if cat and cat.get("cards") and (cat["accuracy"] < 1 or cat["coverage"] < 1):   # категория шла не из БД
        out += ["", "## Шаг 0: категория", "",
                md_table(pd.DataFrame([cat], index=["карточки"]), ["cards", "coverage", "accuracy", "accuracy_covered"])]
    out += ["", "## По атрибутам", "",
            md_table(m.by_attr, ["gold", "accepted", "P", "R", "F1", "text_share", "R_text", "P_any", "R_any"],
                     "attr_id")]
    if len(m.categories):
        out += ["", "## По категориям: топ-20 по трафику, хвост, holdout", "",
                ("traffic — доля кликов категории (core_traffic_weights). Категории holdout_category целиком "
                 "отложены как новые и в топ и хвост не входят."), "",
                md_table(m.categories, ["traffic", "gold", "accepted", "P", "R", "F1", "R_text"], "категория")]
    for col, title in SLICES.items():
        if col in m.slices:
            out += ["", f"## Срез: {title}", "", md_table(m.slices[col], ["gold", "accepted", "P", "R", "F1", "R_text"],
                                                           col)]
    if len(m.steps):
        out += ["", "## Какой шаг дал принятое значение", "", md_table(m.steps, ["rows", "share", "gold", "P"], "step")]
    if len(m.status):
        out += ["", "## Статусы", "", md_table(m.status.to_frame("share"), ["share"], "status")]
    return "\n".join(out) + "\n"
