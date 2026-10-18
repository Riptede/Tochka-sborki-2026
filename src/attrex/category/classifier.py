"""Шаг 0 каскада — категория карточки (b2).

У 79% карточек нет оффера, а значит, нет категории и схемы (EDA, раздел 9). Порядок источников:
1. категория из БД: categoryId фида или sku_seo_id кликов (core_cards.category_source = feed | clicks);
2. иначе — по заголовку: первые 1–2 слова (без «восстановленный», «уценённый») → самая частая категория среди
   заголовков train. Базлайн EDA: покрытие 99.7%, accuracy 93.5%.

Уверенность — доля самой частой категории среди заголовков с тем же ключом, со сглаживанием по числу примеров:
у ключа из двух заголовков «2 из 2» не значит 100%. Ниже category_min_conf пайплайн берёт только сквозные атрибуты.
"""
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace

from attrex.contracts import BuildContext, Card, CategorySource
from attrex.normalize.dictionary import top_vote
from attrex.normalize.text import phrase_key

_CONDITION_PREFIX = re.compile(r"^(восстановленн\w*|уцененн\w*)\s+")
_ROWS_SQL = """
    SELECT c.title, c.category_id
    FROM core_cards c JOIN core_splits s USING (sku_id)
    WHERE s.fold IN (SELECT unnest(?)) AND c.category_source = 'feed' AND c.title IS NOT NULL
"""


def title_key(title: str, n_words: int) -> str:
    """Первые слова заголовка, по которым угадываем категорию: «Восстановленный смартфон Apple» → «смартфон apple»."""
    words = _CONDITION_PREFIX.sub("", phrase_key(title)).split()
    return " ".join(words[:n_words])


@dataclass(frozen=True)
class CategoryGuess:
    category_id: int
    confidence: float
    key: str


class TitleCategoryClassifier:
    """Ключ из первых слов заголовка → самая частая категория. Сначала по двум словам, где их нет — по одному."""

    def __init__(self, tables: dict[int, dict[str, tuple[int, int, int]]], prior: float = 1.0):
        self._tables = tables          # число слов → ключ → (категория, голоса за неё, всего голосов)
        self.prior = prior
        self.lengths = sorted(tables, reverse=True)

    @classmethod
    def learn(cls, rows: Iterable[tuple[str, int]], lengths: Iterable[int] = (2, 1), prior: float = 1.0
              ) -> "TitleCategoryClassifier":
        lengths = tuple(lengths)
        counts: dict[int, dict[str, Counter]] = {n: defaultdict(Counter) for n in lengths}
        for title, category_id in rows:
            for n in lengths:
                key = title_key(title, n)
                if key:
                    counts[n][key][category_id] += 1
        tables = {}
        for n, by_key in counts.items():
            tables[n] = {}
            for key, c in by_key.items():
                cat, top = top_vote(c)
                tables[n][key] = (cat, top, sum(c.values()))
        return cls(tables, prior)

    @classmethod
    def from_context(cls, ctx: BuildContext) -> "TitleCategoryClassifier":
        return cls.learn(ctx.con.execute(_ROWS_SQL, [list(ctx.knowledge_folds)]).fetchall())

    def predict(self, title: str) -> CategoryGuess | None:
        for n in self.lengths:
            key = title_key(title, n)
            hit = self._tables[n].get(key)
            if hit:
                cat, top, total = hit
                return CategoryGuess(cat, round(top / (total + self.prior), 4), key)
        return None


class CategoryResolver:
    """Категория из БД остаётся как есть; карточке без неё — категория по заголовку."""

    def __init__(self, classifier: TitleCategoryClassifier):
        self.classifier = classifier

    def resolve(self, card: Card) -> Card:
        if card.category_id is not None:
            return card
        guess = self.classifier.predict(card.title)
        if guess is None:
            return card
        return replace(card, category_id=guess.category_id, category_conf=guess.confidence,
                       category_source=CategorySource.TITLE)


def build(ctx: BuildContext) -> CategoryResolver:
    return CategoryResolver(ctx.resource(TitleCategoryClassifier.from_context))
