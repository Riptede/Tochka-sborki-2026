"""Оркестратор каскада (b4): карточка → категория → схема → шаги → уверенность → решение → Prediction.

Шаги и их порядок — configs/pipeline.yaml, раздел cascade: модуль шага отдаёт build(ctx) -> Extractor.
Новый шаг подключается строкой в конфиге, этот файл не меняется. Ablation — прогон с подмножеством шагов.

Решение по атрибуту (шаг 8): accepted — значение подтверждено текстом этой карточки (grounding), уверенность
не ниже порога атрибута и это не значение другой карточки той же модели; review — остальные значения; empty — нет.
"""
import importlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml

from attrex.confidence import aggregate
from attrex.confidence.aggregate import Aggregator, AttrDecision
from attrex.config import validate_pipeline
from attrex.contracts import BuildContext, Card, CategorySchema, Extractor, Prediction, Source, Status, Step, Usage
from attrex.normalize.grounding import Grounding
from attrex.normalize.resources import NormalizeResources
from attrex.paths import CONFIGS
from attrex.registry import Registry, load_registry

_SCHEMA_SQL = "SELECT category_id, attr_id, inherited_from FROM core_category_schema WHERE required ORDER BY fill_rate DESC"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    with open(path or CONFIGS / "pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_pipeline(cfg, Path(path or CONFIGS / "pipeline.yaml"))
    return cfg


class CategoryResolver(Protocol):
    """Шаг 0: дополняет карточку категорией и её уверенностью, если их нет."""

    def resolve(self, card: Card) -> Card: ...


class SchemaProvider:
    """Шаг 1: обязательные атрибуты категории из core_category_schema (b1).

    Категория неизвестна или неуверенна (category_conf < category_min_conf) — только сквозные атрибуты (EDA, раздел 9).
    """

    def __init__(self, required: dict[int, tuple[str, ...]], inherited: dict[int, int | None],
                 global_attrs: Sequence[str], fallback: Sequence[str], min_conf: float):
        self.required, self.inherited = required, inherited
        self.global_attrs = frozenset(global_attrs)
        self.fallback = tuple(fallback)
        self.min_conf = min_conf

    @classmethod
    def from_context(cls, ctx: BuildContext) -> "SchemaProvider":
        required: dict[int, list[str]] = defaultdict(list)
        inherited: dict[int, int | None] = {}
        for cid, attr_id, parent in ctx.con.execute(_SCHEMA_SQL).fetchall():
            required[cid].append(attr_id)
            inherited[cid] = parent
        cfg = ctx.config
        return cls({c: tuple(a) for c, a in required.items()}, inherited, load_registry().global_attrs(),
                   cfg["schema"]["fallback_required"], cfg["confidence"]["category_min_conf"])

    def for_card(self, card: Card) -> CategorySchema:
        cid = card.category_id
        if cid is None or cid not in self.required:
            return CategorySchema(cid, self.fallback, global_only=True)
        req = self.required[cid]
        if card.category_conf is not None and card.category_conf < self.min_conf:
            return CategorySchema(cid, tuple(a for a in req if a in self.global_attrs) or self.fallback,
                                  self.inherited.get(cid), global_only=True)
        return CategorySchema(cid, req, self.inherited.get(cid))


class Pipeline:
    def __init__(self, steps: Sequence[Extractor], aggregator: Aggregator, grounding: Grounding,
                 schemas: SchemaProvider, registry: Registry, resolver: CategoryResolver | None = None):
        self.steps = tuple(steps)
        self.aggregator, self.grounding, self.schemas = aggregator, grounding, schemas
        self.registry, self.resolver = registry, resolver

    @property
    def step_names(self) -> list[str]:
        return [str(s.step) for s in self.steps]

    def run(self, card: Card) -> list[Prediction]:
        """Одна строка на каждый обязательный атрибут схемы карточки, включая пустые."""
        if self.resolver is not None:
            card = self.resolver.resolve(card)
        schema = self.schemas.for_card(card)
        candidates = []
        for step in self.steps:
            candidates += step.extract(card, schema, candidates)
        decisions = self.aggregator.decide(candidates, schema.required)
        return [self._predict(card, decisions[a]) for a in schema.required]

    def drain_usage(self) -> Usage:
        total = Usage()
        for step in self.steps:
            drain = getattr(step, "drain_usage", None)
            if drain is not None:
                total = total + drain()
        return total

    def _predict(self, card: Card, d: AttrDecision) -> Prediction:
        base = {"sku_id": card.sku_id, "category_id": card.category_id, "category_conf": card.category_conf,
                "attr_id": d.attr_id}
        if d.empty:
            return Prediction(**base, value=None, unit=None, confidence=None, status=Status.EMPTY,
                              step=Step.NONE, source=Source.NONE)
        grounded = [c for c in d.support if self.grounding.check(card, c)]
        best = max(grounded, key=lambda c: c.score) if grounded else d.best
        accepted = (bool(grounded) and best.step is not Step.MODEL_MATCH
                    and d.confidence >= self.aggregator.threshold(d.attr_id))
        spec = self.registry[d.attr_id]
        return Prediction(**base, value=d.value, unit=spec.unit if spec.kind == "number" else None,
                          confidence=d.confidence, status=Status.ACCEPTED if accepted else Status.REVIEW,
                          step=best.step, source=best.source, value_raw=best.value_raw, evidence=best.evidence)


def _factory(module: str) -> Callable[[BuildContext], object]:
    return importlib.import_module(module).build


def build_pipeline(ctx: BuildContext, steps: Sequence[str] | None = None) -> Pipeline:
    """Пайплайн из шагов каскада configs/pipeline.yaml; steps — подмножество (порядок всегда как в каскаде)."""
    cascade: dict[str, str] = ctx.config["cascade"]
    names = list(steps) if steps else list(cascade)
    unknown = set(names) - set(cascade)
    if unknown:
        raise ValueError(f"нет в cascade (configs/pipeline.yaml): {sorted(unknown)}")
    extractors = []
    for name in (n for n in cascade if n in names):
        ex = _factory(cascade[name])(ctx)
        if ex.step is not Step(name):
            raise ValueError(f"модуль {cascade[name]} отдаёт шаг {ex.step}, а в каскаде он {name}")
        extractors.append(ex)
    resolver_module = (ctx.config.get("category") or {}).get("resolver")
    resolver = _factory(resolver_module)(ctx) if resolver_module else None
    res = ctx.resource(NormalizeResources.from_context)
    return Pipeline(extractors, aggregate.build(ctx), res.grounding, ctx.resource(SchemaProvider.from_context),
                    res.registry, resolver)
