"""Общие ресурсы нормализатора: справочник, нормализатор значений и grounding — один набор на все шаги."""
from collections.abc import Iterable
from dataclasses import dataclass

from attrex.contracts import BuildContext
from attrex.normalize.dictionary import RefDictionary, canon_offers_of
from attrex.normalize.grounding import Grounding
from attrex.normalize.keys import KEY_FUNCTIONS
from attrex.normalize.settings import NormalizeSettings, load_settings
from attrex.normalize.values import ValueNormalizer
from attrex.registry import Registry, load_registry


@dataclass(frozen=True)
class NormalizeResources:
    registry: Registry
    settings: NormalizeSettings
    dictionary: RefDictionary
    normalizer: ValueNormalizer
    grounding: Grounding

    @classmethod
    def create(cls, registry: Registry, settings: NormalizeSettings, dictionary: RefDictionary) -> "NormalizeResources":
        normalizer = ValueNormalizer(registry, dictionary, settings.scores, settings.key_functions)
        return cls(registry, settings, dictionary, normalizer, Grounding(normalizer))

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[str, str, str, int]], registry: Registry | None = None,
                  settings: NormalizeSettings | None = None, filter_canons: bool = False) -> "NormalizeResources":
        """Справочник из строк (attr_id, написание, канон, голоса) — для тестов и экспериментов без БД.

        canon_filters рассчитаны на объём train (≥ 20 офферов), поэтому в маленьких фикстурах по умолчанию выключены.
        """
        registry, settings, rows = registry or load_registry(), settings or load_settings(), list(rows)
        filters = settings.canon_filters if filter_canons else None
        dictionary = RefDictionary.from_rows([*rows, *settings.synonym_rows()], key_fns(settings), settings.inflected,
                                             settings.dictionary_min_support, filters, canon_offers_of(rows))
        return cls.create(registry, settings, dictionary)

    @classmethod
    def from_context(cls, ctx: BuildContext) -> "NormalizeResources":
        registry, settings = load_registry(), load_settings()
        attrs = [a for a, s in registry.attrs.items() if s.kind == "category"]
        dictionary = RefDictionary.from_db(ctx.con, ctx.knowledge_folds, attrs, settings.synonym_rows(),
                                           key_fns(settings), settings.inflected, settings.dictionary_min_support,
                                           settings.canon_filters)
        return cls.create(registry, settings, dictionary)


def key_fns(settings: NormalizeSettings) -> dict:
    return {a: KEY_FUNCTIONS[name] for a, name in settings.key_functions.items()}
