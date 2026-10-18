"""Реестр атрибутов из configs/attributes.yaml.

attr_id в коде, БД, predictions и промптах берётся только отсюда.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from attrex.config import validate_attributes
from attrex.paths import CONFIGS

KINDS = {"number", "category", "text"}


@dataclass(frozen=True)
class AttrSpec:
    attr_id: str
    name_ru: str
    kind: str
    unit: str | None = None
    multi: bool = False
    feed_field: str | None = None
    feed_facet: tuple[str, ...] = ()
    feed_search: tuple[str, ...] = ()
    scope: str | tuple[int, ...] = "global"

    @property
    def is_global(self) -> bool:
        return self.scope == "global"


@dataclass(frozen=True)
class Registry:
    attrs: dict[str, AttrSpec]
    required: dict[int, tuple[str, ...]]
    placeholders: frozenset[str]
    service_fields: frozenset[str]
    category_names: dict[int, str] = field(default_factory=dict)

    def __getitem__(self, attr_id: str) -> AttrSpec:
        return self.attrs[attr_id]

    def global_attrs(self) -> list[str]:
        return [a for a, s in self.attrs.items() if s.is_global]

    def is_placeholder(self, value: str | None) -> bool:
        return value is None or value.strip().lower() in self.placeholders

    def by_feed_name(self) -> dict[str, str]:
        """Имя поля фида (фасет или _search) → attr_id."""
        out = {}
        for a, s in self.attrs.items():
            for name in (*s.feed_facet, *s.feed_search):
                out[name] = a
        return out


def _spec(attr_id: str, d: dict) -> AttrSpec:
    kind = d["kind"]
    if kind not in KINDS:
        raise ValueError(f"{attr_id}: неизвестный kind {kind!r}")
    if kind == "number" and not d.get("unit"):
        raise ValueError(f"{attr_id}: у числового атрибута должна быть каноническая единица")
    feed = d.get("feed", {})
    scope = d.get("scope", "global")
    return AttrSpec(
        attr_id=attr_id, name_ru=d["name_ru"], kind=kind, unit=d.get("unit"), multi=bool(d.get("multi", False)),
        feed_field=feed.get("field"), feed_facet=tuple(feed.get("facet", ())), feed_search=tuple(feed.get("search", ())),
        scope=scope if scope == "global" else tuple(int(c) for c in scope),
    )


@lru_cache(maxsize=1)
def load_registry(path: str | None = None) -> Registry:
    with open(path or CONFIGS / "attributes.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_attributes(cfg, Path(path or CONFIGS / "attributes.yaml"))
    attrs = {a: _spec(a, d) for a, d in cfg["attributes"].items()}
    required = {}
    for cid, c in cfg.get("categories", {}).items():
        unknown = set(c["required"]) - set(attrs)
        if unknown:
            raise ValueError(f"категория {cid}: нет в реестре {sorted(unknown)}")
        required[int(cid)] = tuple(c["required"])
    return Registry(
        attrs=attrs,
        required=required,
        placeholders=frozenset(v.lower() for v in cfg.get("placeholders", [])),
        service_fields=frozenset(cfg.get("service_fields", [])),
        category_names={int(cid): c["name_ru"] for cid, c in cfg.get("categories", {}).items()},
    )
