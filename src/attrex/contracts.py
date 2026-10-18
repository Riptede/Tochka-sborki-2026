"""Контракты между блоками. Меняем только по договорённости всей команды (см. CLAUDE.md).

Card           — вход пайплайна: текст карточки.
CategorySchema — какие атрибуты нужно заполнить у карточки этой категории.
Candidate      — значение, которое предложил один шаг каскада (правила, retrieval, LLM).
Extractor      — интерфейс шага каскада: Card + схема + кандидаты прошлых шагов → новые кандидаты.
BuildContext   — что получает фабрика шага `build(ctx) -> Extractor` при сборке пайплайна.
Prediction     — итоговое решение по атрибуту карточки; строка файла predictions.
"""
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    import duckdb

T = TypeVar("T")


class Step(StrEnum):
    """Шаг каскада, который дал значение."""
    RULES = "rules"            # regex, словари, правила по заголовку
    PARSE = "parse"            # парсер блоков «ключ: значение» в описании
    RETRIEVAL = "retrieval"    # e5 + справочник значений
    LLM = "llm"                # LLM со схемой и кандидатами
    MODEL_MATCH = "model_match"  # значение из карточки той же модели; вне основной метрики
    NONE = "none"              # значение не найдено


class Source(StrEnum):
    """Где в карточке найдено подтверждение."""
    TITLE = "title"
    DESCRIPTION = "description"
    SELLER_ATTRS = "seller_attrs"
    OTHER_CARD = "other_card"
    NONE = "none"


class Status(StrEnum):
    ACCEPTED = "accepted"      # уверенность ≥ порога, идёт в каталог
    REVIEW = "review"          # ниже порога, ручная проверка
    EMPTY = "empty"            # значения нет


class CategorySource(StrEnum):
    FEED = "feed"              # categoryId оффера
    CLICKS = "clicks"          # sku_seo_id из query_clicks
    TITLE = "title"            # классификатор по заголовку


@dataclass
class Card:
    sku_id: int
    title: str
    description: str = ""
    seller_attrs: dict[str, str] = field(default_factory=dict)
    category_id: int | None = None
    category_conf: float | None = None
    category_source: CategorySource | None = None


@dataclass(frozen=True)
class CategorySchema:
    category_id: int | None
    required: tuple[str, ...]            # attr_id из реестра
    inherited_from: int | None = None    # категория-родитель, если своя схема пустая
    global_only: bool = False            # категория неуверенная: заполняем только сквозные атрибуты


@dataclass
class Candidate:
    attr_id: str
    value: str                 # каноническое значение: число как строка в unit реестра или значение справочника
    step: Step
    source: Source
    score: float               # сырой скор шага, до калибровки
    value_raw: str = ""        # как записано в тексте
    evidence: str = ""         # фрагмент текста, подтверждающий значение (для grounding и UI)


class Extractor(Protocol):
    """Шаг каскада. Каждый шаг (правила, парсер, retrieval, LLM) реализует этот интерфейс.

    prior — кандидаты всех предыдущих шагов по этой карточке. Шаг сам решает, какие атрибуты ему
    обрабатывать: дорогие шаги берут только пустые и неуверенные. Шаг не выбирает итоговое значение
    и не пишет в БД: это делают confidence (b2) и pipeline (b4).

    Платный шаг (LLM, эмбеддинги) дополнительно отдаёт drain_usage() -> Usage: расход с прошлого вызова,
    pipeline складывает его в pred_runs.
    """
    step: Step

    def extract(self, card: Card, schema: CategorySchema, prior: list[Candidate]) -> list[Candidate]: ...


@dataclass
class Usage:
    """Расход платных вызовов (рубли — по ценам configs/pipeline.yaml). Складывается по шагам и процессам."""
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_cost_rub: float = 0.0
    embed_tokens: int = 0
    embed_cost_rub: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(**{k: getattr(self, k) + getattr(other, k) for k in asdict(self)})


@dataclass
class BuildContext:
    """Контекст сборки шагов каскада. Каждый модуль шага отдаёт фабрику `build(ctx) -> Extractor`.

    Словари и индексы шаг строит из БД через resource(): ресурс строится один раз и общий для всех шагов,
    поэтому правила, парсер и grounding работают с одним справочником. Готовый шаг не держит соединение с БД:
    его можно передать в другой процесс.
    """
    con: "duckdb.DuckDBPyConnection"                  # БД только на чтение
    config: dict[str, Any]                            # configs/pipeline.yaml
    knowledge_folds: tuple[str, ...] = ("train",)     # из эталона каких фолдов строим словари; test — никогда
    _resources: dict = field(default_factory=dict, repr=False)

    def resource(self, factory: Callable[["BuildContext"], T]) -> T:
        if factory not in self._resources:
            self._resources[factory] = factory(self)
        return self._resources[factory]


@dataclass
class Prediction:
    sku_id: int
    category_id: int | None
    attr_id: str
    value: str | None
    unit: str | None
    confidence: float | None   # откалиброванная вероятность, что value верно
    status: Status
    step: Step
    source: Source
    value_raw: str = ""
    evidence: str = ""
    category_conf: float | None = None

    def to_row(self) -> dict:
        row = asdict(self)
        for k in ("status", "step", "source"):
            row[k] = str(row[k])
        return row


# Порядок колонок файла outputs/predictions.parquet (и .csv для сдачи)
PREDICTION_COLUMNS = [
    "sku_id", "category_id", "category_conf", "attr_id", "value", "unit", "confidence",
    "status", "step", "source", "value_raw", "evidence",
]
