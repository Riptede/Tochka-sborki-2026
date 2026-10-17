"""Контракты между блоками. Меняем только по договорённости всей команды (см. CLAUDE.md).

Card       — вход пайплайна: текст карточки.
Candidate  — значение, которое предложил один шаг каскада (правила, retrieval, LLM).
Prediction — итоговое решение по атрибуту карточки; строка файла predictions.
"""
from dataclasses import asdict, dataclass, field
from enum import StrEnum


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


@dataclass
class Candidate:
    attr_id: str
    value: str                 # каноническое значение: число как строка в unit реестра или значение справочника
    step: Step
    source: Source
    score: float               # сырой скор шага, до калибровки
    value_raw: str = ""        # как записано в тексте
    evidence: str = ""         # фрагмент текста, подтверждающий значение (для grounding и UI)


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
