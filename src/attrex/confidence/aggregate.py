"""Шаг 5 каскада — уверенность (b2): из кандидатов всех шагов — значение атрибута и его уверенность.

Кандидаты с одним каноном складываются: независимые подтверждения (заголовок и описание, правила и парсер)
повышают уверенность, конкурирующие значения её понижают:

    support(v)    = 1 − Π (1 − w_step · score)   по одному лучшему кандидату на (шаг, источник) со значением v
    confidence(v) = support(v) · support(v) / Σ support(всех значений)

Это сырая уверенность. Калибратор (calibration.py) переводит её в вероятность, что значение верно, отдельно
по атрибуту и шагу, который дал значение. Пороги «в каталог» по атрибуту подбирает та же калибровка.
"""
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from attrex.confidence.calibration import IsotonicCalibrator
from attrex.contracts import BuildContext, Candidate, Step
from attrex.normalize.resources import NormalizeResources
from attrex.paths import CONFIGS

SameFn = Callable[[str, str, str], bool]     # (attr_id, канон, канон) → одно ли значение


@dataclass(frozen=True)
class AttrDecision:
    attr_id: str
    value: str | None
    confidence: float
    best: Candidate | None = None             # кандидат с самым сильным подтверждением: его evidence идёт в файл
    support: tuple[Candidate, ...] = ()       # все кандидаты за это значение
    n_values: int = 0                         # сколько разных значений предлагали шаги

    @property
    def empty(self) -> bool:
        return self.value is None


class Calibrator(Protocol):
    def __call__(self, attr_id: str, step: Step, raw: float) -> float: ...


def identity_calibrator(attr_id: str, step: Step, raw: float) -> float:
    return raw


def _exact(attr_id: str, a: str, b: str) -> bool:
    return a == b


class Aggregator:
    def __init__(self, step_weights: Mapping[Step, float] | None = None, same: SameFn = _exact,
                 calibrator: Calibrator = identity_calibrator, accept_threshold: float = 0.75,
                 thresholds: Mapping[str, float] | None = None):
        self.step_weights = {Step(k): float(v) for k, v in (step_weights or {}).items()}
        self.same = same
        self.calibrator = calibrator
        self.accept_threshold = accept_threshold
        self.thresholds = dict(thresholds or {})

    def threshold(self, attr_id: str) -> float:
        return self.thresholds.get(attr_id, self.accept_threshold)

    def decide(self, candidates: Iterable[Candidate], attrs: Iterable[str]) -> dict[str, AttrDecision]:
        """Решение по каждому атрибуту из attrs; атрибут без кандидатов — пустое решение."""
        by_attr: dict[str, list[Candidate]] = {}
        for c in candidates:
            by_attr.setdefault(c.attr_id, []).append(c)
        return {a: self._decide_attr(a, by_attr.get(a, [])) for a in attrs}

    def _decide_attr(self, attr_id: str, cands: list[Candidate]) -> AttrDecision:
        groups = self._group(attr_id, cands)
        if not groups:
            return AttrDecision(attr_id, None, 0.0)
        supports = [self._support(g) for g in groups]
        total = sum(supports)
        i = max(range(len(groups)), key=lambda k: supports[k])
        raw = supports[i] * supports[i] / total if total else 0.0
        best = max(groups[i], key=lambda c: self._weight(c) * c.score)
        # значение, которое подтвердил LLM, калибруется своей кривой «attr/llm»: подтверждение меняет точность,
        # а кривая правил его не видела и упирается в свой потолок
        step = Step.LLM if any(c.step is Step.LLM for c in groups[i]) else best.step
        return AttrDecision(attr_id, best.value, round(self.calibrator(attr_id, step, raw), 4), best,
                            tuple(groups[i]), len(groups))

    def _group(self, attr_id: str, cands: list[Candidate]) -> list[list[Candidate]]:
        groups: list[list[Candidate]] = []
        for c in sorted(cands, key=lambda c: -c.score):
            for g in groups:
                if self.same(attr_id, g[0].value, c.value):
                    g.append(c)
                    break
            else:
                groups.append([c])
        return groups

    def _support(self, group: list[Candidate]) -> float:
        """Независимые подтверждения: один лучший кандидат на пару (шаг, источник)."""
        best: dict[tuple[Step, str], float] = {}
        for c in group:
            key = (c.step, str(c.source))
            best[key] = max(best.get(key, 0.0), min(1.0, self._weight(c) * c.score))
        miss = 1.0
        for s in best.values():
            miss *= 1.0 - s
        return 1.0 - miss

    def _weight(self, c: Candidate) -> float:
        return self.step_weights.get(c.step, 1.0)


def build(ctx: BuildContext) -> Aggregator:
    """Агрегатор с калибровкой из configs/<confidence.calibration>, если файл есть; иначе — сырая уверенность."""
    cfg = ctx.config.get("confidence", {})
    res = ctx.resource(NormalizeResources.from_context)
    calibrator: Calibrator = identity_calibrator
    thresholds = dict(cfg.get("thresholds") or {})
    path = CONFIGS / cfg["calibration"] if cfg.get("calibration") else None
    if path is not None and path.exists():
        fitted = IsotonicCalibrator.load(path)
        calibrator, thresholds = fitted, {**fitted.thresholds, **thresholds}
    return Aggregator(cfg.get("step_weights"), res.normalizer.same, calibrator, cfg.get("accept_threshold", 0.75),
                      thresholds)
