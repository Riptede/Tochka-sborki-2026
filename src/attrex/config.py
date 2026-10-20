"""Схемы и проверка конфигов configs/*.yaml: одно место, где описано, что в них допустимо.

Конфиг проверяется при загрузке (`load_config`, `load_registry`, `load_settings`, калибратор): опечатка в ключе,
значение вне диапазона, attr_id не из реестра или шаг не из `Step` — ошибка ConfigError с файлом и путём до ключа,
а не KeyError в середине прогона. Проверить все конфиги разом:

    uv run python -m attrex.config
"""
import importlib.util
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from attrex.normalize.units import UNIT_TABLE
from attrex.paths import CONFIGS

Share = Annotated[float, Field(ge=0, le=1)]
Positive = Annotated[int, Field(ge=1)]
Rub = Annotated[float, Field(ge=0)]
STEPS = ("rules", "parse", "retrieval", "llm", "model_match")       # Step из contracts.py без none
FOLDS = ("train", "valid", "test", "holdout_category")
UNITS = ("GB", "in", "mAh", "kg", "cm")                              # канонические единицы реестра


class ConfigError(ValueError):
    pass


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")   # неизвестный ключ — ошибка: опечатка не должна молча игнорироваться


# configs/attributes.yaml ---------------------------------------------------------------------------------------
class FeedSpec(Strict):
    field: str | None = None
    facet: list[str] = []
    search: list[str] = []


class AttrSpecModel(Strict):
    name_ru: str
    kind: Literal["number", "category", "text"]
    unit: Literal[UNITS] | None = None
    multi: bool = False
    feed: FeedSpec = FeedSpec()
    scope: Literal["global"] | list[int] = "global"

    @model_validator(mode="after")
    def _unit(self):
        if (self.kind == "number") != (self.unit is not None):
            raise ValueError("единица (unit) обязательна у числового атрибута и запрещена у остальных")
        return self


class CategoryModel(Strict):
    name_ru: str
    required: list[str] = Field(min_length=1)


class AttributesConfig(Strict):
    attributes: dict[str, AttrSpecModel] = Field(min_length=1)
    categories: dict[int, CategoryModel] = {}
    placeholders: list[str] = []
    service_fields: list[str] = []

    @model_validator(mode="after")
    def _required_known(self):
        for cid, c in self.categories.items():
            _known(c.required, self.attributes, f"categories.{cid}.required")
        return self


# configs/pipeline.yaml -----------------------------------------------------------------------------------------
class SchemaSection(Strict):
    required_fill_rate: Share
    min_offers_for_own_schema: Positive
    fallback_required: list[str] = Field(min_length=1)


class ConfidenceSection(Strict):
    accept_threshold: Share
    category_min_conf: Share
    thresholds: dict[str, Share] = {}
    calibration: str | None = None
    step_weights: dict[Literal[STEPS], Annotated[float, Field(ge=0)]] = {}

    @field_validator("calibration")
    @classmethod
    def _calibration_exists(cls, v):
        if v is not None and not (CONFIGS / v).exists():
            raise ValueError(f"нет файла configs/{v}: сгенерируйте его python -m attrex.confidence.calibration")
        return v


class LlmSection(Strict):
    enabled: bool = False
    model: str
    fallback_model: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_cards_per_call: Positive = 1
    self_consistency_samples: Positive = 1
    budget_rub: Rub = 0
    temperature: Annotated[float, Field(ge=0, le=2)] = 0
    max_tokens: Positive = 600
    timeout_s: Annotated[float, Field(gt=0)] = 60
    max_retries: Annotated[int, Field(ge=0)] = 2
    max_description_chars: Positive = 2000
    max_options: Positive = 30
    attrs: list[str] | None = None         # каким атрибутам можно в LLM; None — всем обязательным
    require_candidate: bool = False        # только атрибуты, где прошлые шаги нашли кандидата (LLM-проверка)
    context_chars: Annotated[int, Field(ge=0)] = 0   # >0: из описания — окна ±N символов вокруг зацепок
    score: Share = 0.85
    cache: str | None = None
    prices_rub_per_1k: dict[str, tuple[Rub, Rub]]

    @model_validator(mode="after")
    def _priced(self):
        # без цены не посчитать ни бюджет, ни стоимость прогона: такую модель не пускаем
        for m in (self.model, self.fallback_model):
            if m and m not in self.prices_rub_per_1k:
                raise ValueError(f"нет цены модели {m} в prices_rub_per_1k")
        if self.enabled and self.budget_rub <= 0:
            raise ValueError("llm.enabled: true требует budget_rub > 0")
        return self


class EmbeddingsSection(Strict):
    backend: Literal["local", "api"]
    model: str
    local_model: str
    batch_size: Positive = 64
    budget_rub: Rub = 0
    prices_rub_per_1k: dict[str, Rub] = {}


class SplitSection(Strict):
    seed: int
    test_share: Share
    valid_share: Share
    holdout_category_share: Share

    @model_validator(mode="after")
    def _sum(self):
        if self.test_share + self.valid_share + self.holdout_category_share >= 1:
            raise ValueError("на train ничего не остаётся: сумма долей test, valid и holdout должна быть < 1")
        return self


class ReviewSection(Strict):
    traffic_weight_scale: Annotated[float, Field(ge=0)] = 1000   # множитель веса по трафику в приоритете очереди


class CategorySection(Strict):
    resolver: str | None = None


class PipelineConfig(Strict):
    cascade: dict[Literal[STEPS], str] = Field(min_length=1)
    category: CategorySection = CategorySection()
    knowledge_folds: list[Literal["train", "valid"]] = ["train"]   # test и holdout_category — никогда
    schema_: SchemaSection = Field(alias="schema")
    confidence: ConfidenceSection
    llm: LlmSection
    embeddings: EmbeddingsSection
    split: SplitSection
    review: ReviewSection = ReviewSection()

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("cascade")
    @classmethod
    def _modules_exist(cls, v: dict[str, str]):
        for step, module in v.items():
            _importable(module, f"cascade.{step}")
        return v

    @field_validator("category")
    @classmethod
    def _resolver_exists(cls, v: CategorySection):
        if v.resolver:
            _importable(v.resolver, "category.resolver")
        return v


# configs/normalize.yaml (b2): верхний уровень и ссылки на attr_id; вложенные параметры шагов читают сами шаги ---
class CanonFilterSection(Strict):
    min_offers: Positive = 1
    exclude: list[str] = []
    more_frequent_in: list[str] = []

    @field_validator("exclude", mode="before")
    @classmethod
    def _as_str(cls, v):
        return [str(x) for x in v or ()]


class RulesSection(Strict):
    """Множители уверенности и окна правил (normalize/rules.py, RuleSettings)."""
    offset_penalty: Share = 0.85
    no_prefix_penalty: Share = 0.8
    model_factor: Share = 0.8
    unanchored_factor: Share = 0.9
    window_before: Positive = 30
    window_after: Positive = 8
    after_penalty: Positive = 2


class NormalizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: dict[str, list[str]] = {}
    unit_consumers: dict[str, Literal[UNITS]] = {}
    aliases: dict[str, list[str]] = {}
    key_functions: dict[str, str] = {}
    inflected: list[str] = []
    dictionary_min_support: Positive = 3
    canon_filters: dict[str, CanonFilterSection] = {}
    anchored_units: list[str] = []
    title_tail_attrs: list[str] = []
    manual_weight: Positive = 20
    review_synonyms: str | None = None
    synonyms: dict[str, dict[str, str]] = {}
    compat_words: list[str] = []
    scores: dict[Literal["title", "description", "unknown_value", "partial_match", "assumed", "title_inner"],
                 Share] = {}
    parse: dict[str, Any] = {}
    rules: RulesSection = RulesSection()
    retrieval: dict[str, Any] = {}
    model_match: dict[str, Any] = {}
    title_layout: dict[str, Any] = {}
    dimensions: dict[str, Any] = {}

    @field_validator("synonyms", mode="before")
    @classmethod
    def _synonyms_as_str(cls, v):
        # YAML читает «0: ...» и «true: ...» не строками; в словаре синонимов ключ и значение — всегда текст
        return {a: {str(k): str(x) for k, x in (s or {}).items()} for a, s in (v or {}).items()}


# configs/calibration.yaml (генерирует b2) --------------------------------------------------------------------------
class CurveModel(Strict):
    n: int = Field(ge=0)
    x: list[float]
    y: list[Share]

    @model_validator(mode="after")
    def _shape(self):
        if len(self.x) != len(self.y) or not self.x:
            raise ValueError("у кривой x и y должны быть непустыми и одной длины")
        if any(b < a for a, b in zip(self.x, self.x[1:])) or any(b < a for a, b in zip(self.y, self.y[1:])):
            raise ValueError("кривая калибровки должна быть неубывающей по x и y")
        return self


class CalibrationConfig(Strict):
    meta: dict[str, Any] = {}
    thresholds: dict[str, Share] = {}
    curves: dict[str, CurveModel] = {}


# проверка ----------------------------------------------------------------------------------------------------------
def _known(attr_ids: Iterable[str], attributes: Iterable[str], where: str) -> None:
    unknown = sorted(set(attr_ids) - set(attributes))
    if unknown:
        raise ValueError(f"{where}: нет в реестре configs/attributes.yaml: {unknown}")


def _importable(module: str, where: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise ValueError(f"{where}: модуль {module} не найден")


def _format(path: Path, e: ValidationError) -> str:
    lines = [f"{path.name}: ошибка в конфиге"]
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(корень)"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)


def check(model: type[BaseModel], cfg: dict, path: Path) -> BaseModel:
    try:
        return model.model_validate(cfg or {})
    except ValidationError as e:
        raise ConfigError(_format(path, e)) from None


def _attr_ids() -> set[str]:
    path = CONFIGS / "attributes.yaml"
    return set(check(AttributesConfig, yaml.safe_load(path.read_text(encoding="utf-8")), path).attributes)


def validate_attributes(cfg: dict, path: Path = CONFIGS / "attributes.yaml") -> None:
    check(AttributesConfig, cfg, path)


def validate_pipeline(cfg: dict, path: Path = CONFIGS / "pipeline.yaml") -> None:
    m = check(PipelineConfig, cfg, path)
    try:
        attrs = _attr_ids()
        _known(m.schema_.fallback_required, attrs, "schema.fallback_required")
        _known(m.confidence.thresholds, attrs, "confidence.thresholds")
    except ValueError as e:
        raise ConfigError(f"{path.name}: {e}") from None


def validate_normalize(cfg: dict, path: Path = CONFIGS / "normalize.yaml") -> None:
    m = check(NormalizeConfig, cfg, path)
    try:
        attrs = _attr_ids()
        for name in ("keywords", "unit_consumers", "aliases", "key_functions", "synonyms"):
            _known(getattr(m, name), attrs, name)
        _known(m.inflected, attrs, "inflected")
        _known(m.retrieval.get("attrs", []), attrs, "retrieval.attrs")
        _known(m.canon_filters, attrs, "canon_filters")
        _known({a for f in m.canon_filters.values() for a in f.more_frequent_in}, attrs, "canon_filters.more_frequent_in")
        _known(m.title_tail_attrs, attrs, "title_tail_attrs")
        unknown_units = sorted(set(m.anchored_units) - set(UNIT_TABLE))
        if unknown_units:
            raise ValueError(f"anchored_units: нет в UNIT_TABLE (normalize/units.py): {unknown_units}")
    except ValueError as e:
        raise ConfigError(f"{path.name}: {e}") from None


def validate_calibration(cfg: dict, path: Path = CONFIGS / "calibration.yaml") -> None:
    m = check(CalibrationConfig, cfg, path)
    try:
        attrs = _attr_ids()
        _known(m.thresholds, attrs, "thresholds")
        _known({k.split("/")[0] for k in m.curves if k != "*"}, attrs, "curves")
    except ValueError as e:
        raise ConfigError(f"{path.name}: {e}") from None


VALIDATORS = {"attributes.yaml": validate_attributes, "pipeline.yaml": validate_pipeline,
              "normalize.yaml": validate_normalize, "calibration.yaml": validate_calibration}


def validate_all(configs: Path = CONFIGS) -> list[str]:
    """Проверить все конфиги; вернуть список ошибок (пустой — всё в порядке)."""
    errors = []
    for name, fn in VALIDATORS.items():
        path = configs / name
        if not path.exists():
            continue
        try:
            fn(yaml.safe_load(path.read_text(encoding="utf-8")), path)
        except ConfigError as e:
            errors.append(str(e))
    return errors


def main() -> None:
    errors = validate_all()
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"конфиги в порядке: {', '.join(n for n in VALIDATORS if (CONFIGS / n).exists())}")


if __name__ == "__main__":
    main()
