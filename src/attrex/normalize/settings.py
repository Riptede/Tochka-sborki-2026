"""Параметры нормализатора из configs/normalize.yaml (блок b2)."""
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from attrex.normalize.text import fold
from attrex.paths import CONFIGS


@dataclass(frozen=True)
class Scores:
    title: float = 0.9
    description: float = 0.6
    unknown_value: float = 0.3
    partial_match: float = 0.8
    assumed: float = 0.7


@dataclass(frozen=True)
class ParseSettings:
    alias_score: float = 0.9          # скор пары с ручным алиасом ключа
    learn_min_support: int = 10       # выученный алиас: сколько карточек с этим ключом и эталоном атрибута
    learn_min_precision: float = 0.7  # и доля, где значение ключа подтвердило эталон


@dataclass(frozen=True)
class NormalizeSettings:
    keywords: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unit_consumers: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    key_functions: dict[str, str] = field(default_factory=dict)
    inflected: tuple[str, ...] = ()
    dictionary_min_support: int = 3
    manual_weight: int = 20
    synonyms: dict[str, dict[str, str]] = field(default_factory=dict)
    compat_words: frozenset[str] = frozenset()
    scores: Scores = Scores()
    parse: ParseSettings = ParseSettings()
    retrieval: dict = field(default_factory=dict)     # разбирает normalize/retrieval.py: RetrievalSettings
    model_match: dict = field(default_factory=dict)   # разбирает normalize/model_match.py: ModelMatchSettings
    min_prefix_count: int = 3
    max_brand_offset: int = 2
    dims_min_support: int = 5
    dims_min_rate: float = 0.5

    def synonym_rows(self) -> list[tuple[str, str, str, int]]:
        """Ручные синонимы в формате строк справочника: (attr_id, написание, канон, голоса)."""
        return [(a, s, c, self.manual_weight) for a, syn in self.synonyms.items() for s, c in syn.items()]


def _folded(words) -> tuple[str, ...]:
    return tuple(fold(str(w)).strip() for w in words or ())


def parse_settings(cfg: dict) -> NormalizeSettings:
    layout, dims = cfg.get("title_layout", {}), cfg.get("dimensions", {})
    return NormalizeSettings(
        keywords={a: _folded(w) for a, w in cfg.get("keywords", {}).items()},
        unit_consumers=dict(cfg.get("unit_consumers", {})),
        aliases={a: _folded(w) for a, w in cfg.get("aliases", {}).items()},
        key_functions=dict(cfg.get("key_functions", {})),
        inflected=tuple(cfg.get("inflected", ())),
        dictionary_min_support=int(cfg.get("dictionary_min_support", 3)),
        manual_weight=int(cfg.get("manual_weight", 20)),
        synonyms={a: {str(k): str(v) for k, v in s.items()} for a, s in cfg.get("synonyms", {}).items()},
        compat_words=frozenset(_folded(cfg.get("compat_words"))),
        scores=Scores(**cfg.get("scores", {})),
        parse=ParseSettings(**cfg.get("parse", {})),
        retrieval=dict(cfg.get("retrieval", {})),
        model_match=dict(cfg.get("model_match", {})),
        min_prefix_count=int(layout.get("min_prefix_count", 3)),
        max_brand_offset=int(layout.get("max_brand_offset", 2)),
        dims_min_support=int(dims.get("min_support", 5)),
        dims_min_rate=float(dims.get("min_rate", 0.5)),
    )


@lru_cache(maxsize=1)
def load_settings(path: str | None = None) -> NormalizeSettings:
    """configs/normalize.yaml и синонимы из решений модераторов (review_synonyms, пишет b5); ручные — важнее."""
    with open(path or CONFIGS / "normalize.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    review = CONFIGS / cfg["review_synonyms"] if cfg.get("review_synonyms") else None
    if review is not None and review.exists():
        learned = (yaml.safe_load(review.read_text(encoding="utf-8")) or {}).get("synonyms") or {}
        manual = cfg.get("synonyms") or {}
        cfg["synonyms"] = {a: {**learned.get(a, {}), **manual.get(a, {})} for a in {*learned, *manual}}
    return parse_settings(cfg)
