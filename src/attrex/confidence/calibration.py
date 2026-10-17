"""Калибровка уверенности (b2): сырая уверенность агрегатора → вероятность, что значение верно.

Изотоническая регрессия по паре (атрибут, шаг); где данных меньше min_n — кривая атрибута, затем общая.
Подбирается на valid прогоном каскада с сырой уверенностью; там же по атрибуту выбирается порог «в каталог» —
наименьший, при котором точность значений с уверенностью не ниже порога ≥ target_precision.
Результат — configs/calibration.yaml (генерируется, руками не правим); агрегатор подхватывает его сам.

    uv run python -m attrex.confidence.calibration --fold valid --workers 2

Калибровка и пороги подбираются на valid, поэтому метрики на valid после неё оптимистичны: итог — на test.
"""
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

GLOBAL = "*"


def isotonic(x: Sequence[float], y: Sequence[float]) -> tuple[list[float], list[float]]:
    """Неубывающая ступенчатая функция по точкам (x, y): pool adjacent violators. Возвращает центры блоков и уровни."""
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    blocks: list[list[float]] = []          # [сумма y, вес, сумма x]
    for xv, yv in zip(xs, ys):
        blocks.append([yv, 1.0, xv])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            y2, w2, x2 = blocks.pop()
            blocks[-1][0] += y2
            blocks[-1][1] += w2
            blocks[-1][2] += x2
    return [float(b[2] / b[1]) for b in blocks], [float(b[0] / b[1]) for b in blocks]


@dataclass(frozen=True)
class Curve:
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    n: int

    def __call__(self, raw: float) -> float:
        return float(np.interp(raw, self.xs, self.ys))

    @classmethod
    def fit(cls, raw: Sequence[float], correct: Sequence[bool]) -> "Curve":
        xs, ys = isotonic(raw, [float(c) for c in correct])
        return cls(tuple(round(v, 4) for v in xs), tuple(round(v, 4) for v in ys), len(raw))

    def __post_init__(self):
        # кривая хранится в YAML: только встроенные float, без типов numpy
        object.__setattr__(self, "xs", tuple(float(v) for v in self.xs))
        object.__setattr__(self, "ys", tuple(float(v) for v in self.ys))


class IsotonicCalibrator:
    """Кривые по ключам «attr/step», «attr» и «*»; берётся самая частная из выученных."""

    def __init__(self, curves: dict[str, Curve], thresholds: dict[str, float] | None = None, meta: dict | None = None):
        self.curves, self.thresholds, self.meta = curves, dict(thresholds or {}), dict(meta or {})

    def __call__(self, attr_id: str, step, raw: float) -> float:
        for key in (f"{attr_id}/{step}", attr_id, GLOBAL):
            curve = self.curves.get(key)
            if curve is not None:
                return curve(raw)
        return raw

    @classmethod
    def fit(cls, rows: Iterable[tuple[str, str, float, bool]], min_n: int = 50) -> "IsotonicCalibrator":
        """rows: (attr_id, step, сырая уверенность, верно ли значение)."""
        groups: dict[str, tuple[list[float], list[bool]]] = defaultdict(lambda: ([], []))
        for attr_id, step, raw, correct in rows:
            for key in (f"{attr_id}/{step}", attr_id, GLOBAL):
                groups[key][0].append(raw)
                groups[key][1].append(correct)
        return cls({k: Curve.fit(x, y) for k, (x, y) in groups.items() if len(x) >= min_n or k == GLOBAL})

    def save(self, path: Path) -> None:
        doc = {"meta": self.meta, "thresholds": {k: round(v, 4) for k, v in sorted(self.thresholds.items())},
               "curves": {k: {"n": c.n, "x": list(c.xs), "y": list(c.ys)} for k, c in sorted(self.curves.items())}}
        header = "# Сгенерировано: python -m attrex.confidence.calibration. Руками не править — перегенерировать.\n"
        path.write_text(header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        curves = {k: Curve(tuple(c["x"]), tuple(c["y"]), c["n"]) for k, c in (doc.get("curves") or {}).items()}
        return cls(curves, doc.get("thresholds"), doc.get("meta"))


def choose_thresholds(rows: Iterable[tuple[str, float, bool]], target_precision: float, default: float,
                      min_n: int = 30, floor: float = 0.5) -> dict[str, float]:
    """rows: (attr_id, откалиброванная уверенность, верно ли). Порог атрибута — наименьшая уверенность, при которой
    точность всех значений не ниже неё ≥ target_precision, но не ниже floor: значение, которое скорее неверно,
    в каталог не идёт, даже если в среднем по атрибуту точность высокая. Мало данных — порог по умолчанию."""
    by_attr: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for attr_id, conf, correct in rows:
        by_attr[attr_id].append((conf, correct))
    out = {}
    for attr_id, items in by_attr.items():
        if len(items) < min_n:
            continue
        items.sort(key=lambda t: -t[0])
        hits, best = 0, None
        for k, (conf, correct) in enumerate(items, 1):
            hits += correct
            last_of_tie = k == len(items) or items[k][0] < conf      # порог берёт все значения с той же уверенностью
            if last_of_tie and hits / k >= target_precision:
                best = conf
        out[attr_id] = round(max(float(best), floor), 4) if best is not None else round(max(default, items[0][0] + 1e-4), 4)
    return out


def main() -> None:
    import argparse

    import duckdb

    from attrex import paths
    from attrex.confidence.aggregate import identity_calibrator
    from attrex.contracts import BuildContext
    from attrex.metrics.scoring import comparison_same
    from attrex.pipeline import build_pipeline, load_config
    from attrex.runs import RunSpec, git_commit, iter_cards, parse_steps, predict

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", default="valid", help="на каком фолде подбирать (не test и не holdout_category)")
    ap.add_argument("--steps", help="шаги каскада через запятую; по умолчанию все")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--target-precision", type=float, default=0.9, help="точность принятых значений по атрибуту")
    ap.add_argument("--floor", type=float, default=0.5, help="порог не ниже: вероятность < 0.5 — скорее неверно")
    ap.add_argument("--min-n", type=int, default=50, help="меньше значений — кривая уровнем выше")
    ap.add_argument("--db", type=Path, default=paths.DB)
    a = ap.parse_args()
    if a.fold in ("test", "holdout_category"):
        ap.error("калибровку подбираем на train/valid; test и holdout_category — только для итоговых метрик")

    cfg = load_config()
    steps = parse_steps(a.steps, list(cfg["cascade"]))
    knowledge = tuple(cfg["knowledge_folds"])
    con = duckdb.connect(str(a.db), read_only=True)
    try:
        pipeline = build_pipeline(BuildContext(con, cfg, knowledge), steps)
        pipeline.aggregator.calibrator = identity_calibrator        # учимся на сырой уверенности
        spec = RunSpec("calibration", steps, knowledge, folds=(a.fold,), workers=a.workers)
        rows = [r for batch, _ in predict(pipeline, iter_cards(con, spec), spec.workers) for r in batch
                if r["value"] is not None]
        gold: dict[tuple[int, str], list[str]] = defaultdict(list)
        for sku, attr_id, value in con.execute(
                "SELECT g.sku_id, g.attr_id, g.value FROM core_gold g JOIN core_splits s USING (sku_id) "
                "WHERE s.fold = ? AND g.value IS NOT NULL", [a.fold]).fetchall():
            gold[sku, attr_id].append(value)
    finally:
        con.close()

    same = comparison_same()
    data = [(r["attr_id"], r["step"], r["confidence"], any(same(r["attr_id"], r["value"], g)
                                                         for g in gold[r["sku_id"], r["attr_id"]]))
            for r in rows if (r["sku_id"], r["attr_id"]) in gold]
    cal = IsotonicCalibrator.fit(data, a.min_n)
    cal.thresholds = choose_thresholds([(attr, cal(attr, step, raw), ok) for attr, step, raw, ok in data],
                                       a.target_precision, cfg["confidence"]["accept_threshold"], floor=a.floor)
    cal.meta = {"fold": a.fold, "knowledge_folds": list(knowledge), "steps": list(steps), "n": len(data),
                "target_precision": a.target_precision, "floor": a.floor, "git_commit": git_commit(),
                "fitted_at": datetime.now().isoformat(timespec="seconds")}
    out = paths.CONFIGS / cfg["confidence"]["calibration"]
    cal.save(out)
    print(f"{out}: {len(data):,} значений с эталоном, кривых {len(cal.curves)}; пороги: {cal.thresholds}")


if __name__ == "__main__":
    main()
