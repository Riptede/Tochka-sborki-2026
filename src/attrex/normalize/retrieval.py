"""Шаг 4 каскада — retrieval к справочнику (b2): ближайшее каноническое значение для фрагмента текста.

Работает только для справочных атрибутов, которые после правил и парсера пустые или неуверенные. Запросы:
значения из блока «ключ: значение», которых нет в справочнике («Цвет: темно-графитовый»), и слова заголовка.
Значения справочника и запросы кодирует Encoder; совпадение — косинус не ниже min_similarity.

Базлайн — CharNgramEncoder: триграммы символов с хешированием, numpy, без моделей и без сети. Эмбеддинги
платформы (configs/pipeline.yaml, embeddings) подключаются другим Encoder — только через llm/client.py (b3).
"""
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from attrex.contracts import BuildContext, Candidate, Card, CategorySchema, Source, Step
from attrex.normalize.resources import NormalizeResources
from attrex.normalize.text import TextField, phrase_key


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Строка на текст, L2-нормированная."""
        ...


class CharNgramEncoder:
    """Триграммы символов слова с границами («^се», «сер», …) → хеш-вектор размерности dim, TF с логарифмом."""

    def __init__(self, dim: int = 4096, n: int = 3):
        self.dim, self.n = dim, n

    def grams(self, text: str) -> dict[int, float]:
        s = f" {phrase_key(text)} "
        counts: dict[int, float] = {}
        for i in range(len(s) - self.n + 1):
            h = zlib.crc32(s[i:i + self.n].encode()) % self.dim      # crc32, а не hash(): одинаково в любом процессе
            counts[h] = counts.get(h, 0.0) + 1.0
        return {h: 1.0 + np.log(c) for h, c in counts.items()}

    def sparse(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Номера ненулевых координат и их веса (L2-нормированные): у слова их около 15 из dim."""
        g = self.grams(text)
        idx = np.fromiter(g.keys(), dtype=np.int64, count=len(g))
        w = np.fromiter(g.values(), dtype=np.float32, count=len(g))
        norm = float(np.linalg.norm(w))
        return idx, (w / norm if norm else w)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            idx, w = self.sparse(t)
            out[i, idx] = w
        return out


@dataclass(frozen=True)
class RetrievalSettings:
    attrs: tuple[str, ...]
    min_similarity: float = 0.75
    open_below: float = 0.7           # атрибут ищем, если у прошлых шагов нет кандидата со скором не ниже
    max_words: int = 2                # запросы из заголовка — по одному и по два слова
    min_value_support: int = 3        # значения справочника реже — не в индексе: это шум эталона

    @classmethod
    def from_config(cls, cfg: dict, default_attrs: Sequence[str]) -> "RetrievalSettings":
        cfg = dict(cfg or {})
        cfg.pop("encoder", None)
        return cls(attrs=tuple(cfg.pop("attrs", default_attrs)), **cfg)


class ValueIndex:
    """Канонические значения атрибута и их векторы (по столбцам: строка матрицы — координата вектора)."""

    def __init__(self, values: Sequence[str], encoder: Encoder):
        self.values = list(values)
        self.encoder = encoder
        self.by_dim = np.ascontiguousarray(encoder.encode(self.values).T)

    def best(self, texts: Sequence[str]) -> list[tuple[int, float]]:
        """Для каждого запроса — номер ближайшего значения и косинус."""
        if not self.values or not texts:
            return []
        if isinstance(self.encoder, CharNgramEncoder):     # разреженный запрос: складываем только его координаты
            sims = np.stack([w @ self.by_dim[idx] for idx, w in map(self.encoder.sparse, texts)])
        else:
            sims = self.encoder.encode(texts) @ self.by_dim
        best = sims.argmax(axis=1)
        return [(int(i), float(sims[q, i])) for q, i in enumerate(best)]


@dataclass(frozen=True)
class _Query:
    text: str
    source: Source
    evidence: str


class RetrievalExtractor:
    step = Step.RETRIEVAL

    def __init__(self, indexes: dict[str, ValueIndex], encoder: Encoder, settings: RetrievalSettings,
                 res: NormalizeResources):
        self.indexes, self.encoder, self.settings, self.res = indexes, encoder, settings, res

    def extract(self, card: Card, schema: CategorySchema, prior: list[Candidate]) -> list[Candidate]:
        open_attrs = [a for a in schema.required if a in self.indexes and self._is_open(a, prior)]
        if not open_attrs:
            return []
        title = TextField(Source.TITLE, card.title or "")
        title_queries = self._title_queries(title)
        out = []
        for attr_id in open_attrs:
            queries = self._parse_queries(attr_id, prior) + title_queries
            if not queries:
                continue
            hits = self.indexes[attr_id].best([q.text for q in queries])
            scored = [(sim, q, self.indexes[attr_id].values[i]) for (i, sim), q in zip(hits, queries)
                      if sim >= self.settings.min_similarity]
            if scored:
                sim, q, value = max(scored, key=lambda s: s[0])
                base = self.res.settings.scores.title if q.source is Source.TITLE else self.res.settings.scores.description
                out.append(Candidate(attr_id=attr_id, value=value, step=self.step, source=q.source,
                                     score=round(sim * base, 4), value_raw=q.text, evidence=q.evidence))
        return out

    def _is_open(self, attr_id: str, prior: list[Candidate]) -> bool:
        return all(c.score < self.settings.open_below for c in prior if c.attr_id == attr_id)

    def _parse_queries(self, attr_id: str, prior: list[Candidate]) -> list[_Query]:
        """Значения из блока «ключ: значение», которых нет в справочнике: ключ знаем, канон — нет."""
        known = self.res.dictionary.values(attr_id)
        return [_Query(c.value_raw, c.source, c.evidence) for c in prior
                if c.attr_id == attr_id and c.step is Step.PARSE and c.value not in known and c.value_raw]

    def _title_queries(self, title: TextField) -> list[_Query]:
        toks = title.tokens
        out = []
        for n in range(1, self.settings.max_words + 1):
            for i in range(len(toks) - n + 1):
                if any(t.text.isdigit() for t in toks[i:i + n]):
                    continue
                text = title.fragment(toks[i].start, toks[i + n - 1].end)
                out.append(_Query(text, Source.TITLE, text))
        return out


def make_encoder(name: str) -> Encoder:
    if name == "char":
        return CharNgramEncoder()
    raise ValueError(f"неизвестный encoder для retrieval: {name!r} (char; api — через llm/client.py, блок b3)")


def build(ctx: BuildContext) -> RetrievalExtractor:
    res = ctx.resource(NormalizeResources.from_context)
    cfg = res.settings.retrieval
    default_attrs = [a for a, s in res.registry.attrs.items() if s.kind == "category" and a != "brand"]
    settings = RetrievalSettings.from_config(cfg, default_attrs)
    encoder = make_encoder(cfg.get("encoder", "char"))
    indexes = {}
    for attr_id in settings.attrs:
        support = res.dictionary.value_support(attr_id)
        values = sorted(v for v, n in support.items() if n >= settings.min_value_support)
        if values:
            indexes[attr_id] = ValueIndex(values, encoder)
    return RetrievalExtractor(indexes, encoder, settings, res)
