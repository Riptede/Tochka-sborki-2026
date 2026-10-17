"""Справочник значений атрибутов: написание в тексте → каноническое значение.

Строится из эталона фолдов знаний (BuildContext.knowledge_folds, по умолчанию train):
- каноническое значение — само себе написание;
- сырое значение `_search` той же карточки — написание продавца («titanium grey» → «серый»);
- ручные синонимы из configs/normalize.yaml (английские цвета, «б/у»).
Одно написание бывает у разных канонов («мятный» → «зеленый» или «мятный»): выбираем самый частый,
доля его голосов — score совпадения.
"""
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import duckdb

from attrex.normalize.keys import KeyFn
from attrex.normalize.text import Token, phrase_key, stem

MAX_WORDS = 6   # длиннее написания в тексте не ищем

_GOLD_SQL = """
    WITH k AS (SELECT sku_id FROM core_splits WHERE fold IN (SELECT unnest(?)))
    SELECT g.attr_id, g.value AS surface, g.value AS canon, count(*) AS n
    FROM core_gold g JOIN k USING (sku_id)
    WHERE g.value IS NOT NULL AND g.attr_id IN (SELECT unnest(?))
    GROUP BY 1, 2, 3
    UNION ALL
    SELECT g.attr_id, trim(p.value), g.value, count(*)
    FROM core_gold g JOIN k USING (sku_id)
    JOIN raw_params p ON p.sku_id = g.sku_id
    JOIN core_attr_name_map m ON m.param_name = trim(p.param_name) AND m.attr_id = g.attr_id AND m.layer = 'search'
    WHERE g.value IS NOT NULL AND g.attr_id IN (SELECT unnest(?))
    GROUP BY 1, 2, 3
"""


@dataclass(frozen=True)
class DictMatch:
    attr_id: str
    value: str        # канон
    score: float      # доля голосов за канон среди всех канонов этого написания
    support: int      # сколько раз написание встречалось в эталоне
    start: int = -1   # позиция найденного написания в свёрнутом тексте (только у find)
    end: int = -1


def _decide(attr_id: str, votes: Counter) -> DictMatch:
    value, n = votes.most_common(1)[0]
    total = sum(votes.values())
    return DictMatch(attr_id, value, n / total, total)


class RefDictionary:
    def __init__(self, votes: dict[str, dict[str, Counter]], max_words: dict[str, int],
                 key_fns: dict[str, KeyFn] | None = None, inflected: Iterable[str] = ()):
        self._key_fns = dict(key_fns or {})
        self._max_words = max_words
        self._exact = {a: {k: _decide(a, c) for k, c in forms.items()} for a, forms in votes.items()}
        self._stems: dict[str, dict[str, DictMatch]] = {}
        for a in set(inflected) & set(votes) - set(self._key_fns):
            merged: dict[str, Counter] = defaultdict(Counter)
            for k, c in votes[a].items():
                merged[" ".join(map(stem, k.split()))].update(c)
            self._stems[a] = {k: _decide(a, c) for k, c in merged.items()}

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[str, str, str, int]], key_fns: dict[str, KeyFn] | None = None,
                  inflected: Iterable[str] = (), min_support: int = 1) -> "RefDictionary":
        """rows: (attr_id, написание, канон, голоса).

        Написание продавца, которое не совпадает ни с одним каноном, берём только при min_support голосов:
        в сырых значениях фида много шума («листья» как материал). Сами каноны остаются всегда — и редкие бренды тоже.
        """
        key_fns = key_fns or {}
        votes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        canon_keys: dict[str, set[str]] = defaultdict(set)
        words: dict[str, Counter] = defaultdict(Counter)    # attr_id → ключ → длина написания в словах
        for attr_id, surface, canon, n in rows:
            key_fn = key_fns.get(attr_id, phrase_key)
            key = key_fn(surface) if surface else None
            if key and canon:
                votes[attr_id][key][canon] += n
                canon_keys[attr_id].add(key_fn(canon) or "")
                words[attr_id][key] = max(words[attr_id][key], len(phrase_key(surface).split()))
        kept = {a: {k: c for k, c in forms.items() if k in canon_keys[a] or sum(c.values()) >= min_support}
                for a, forms in votes.items()}
        max_words = {a: min(MAX_WORDS, max((words[a][k] for k in forms), default=0)) for a, forms in kept.items()}
        return cls(kept, max_words, key_fns, inflected)

    @classmethod
    def from_db(cls, con: duckdb.DuckDBPyConnection, folds: Sequence[str], attrs: Sequence[str],
                extra_rows: Iterable[tuple[str, str, str, int]] = (), key_fns: dict[str, KeyFn] | None = None,
                inflected: Iterable[str] = (), min_support: int = 1) -> "RefDictionary":
        attrs, folds = list(attrs), list(folds)
        rows = con.execute(_GOLD_SQL, [folds, attrs, attrs]).fetchall()
        return cls.from_rows([*rows, *extra_rows], key_fns, inflected, min_support)

    def attrs(self) -> set[str]:
        return set(self._exact)

    def values(self, attr_id: str) -> set[str]:
        return {m.value for m in self._exact.get(attr_id, {}).values()}

    def lookup(self, attr_id: str, text: str) -> DictMatch | None:
        """Значение целиком — написание из справочника (без поиска внутри текста)."""
        return self._match(attr_id, phrase_key(text).split())

    def find(self, attr_id: str, tokens: Sequence[Token]) -> list[DictMatch]:
        """Все написания в тексте, слева направо; при пересечении выигрывает самое длинное."""
        out, i = [], 0
        while i < len(tokens):
            found = self._longest_at(attr_id, tokens, i)
            if found:
                out.append(found[0])
                i += found[1]
            else:
                i += 1
        return out

    def find_at(self, attr_id: str, tokens: Sequence[Token], i: int) -> DictMatch | None:
        """Самое длинное написание, которое начинается с токена i."""
        found = self._longest_at(attr_id, tokens, i)
        return found[0] if found else None

    def _longest_at(self, attr_id: str, tokens: Sequence[Token], i: int) -> tuple[DictMatch, int] | None:
        for n in range(min(self._max_words.get(attr_id, 0), len(tokens) - i), 0, -1):
            hit = self._match(attr_id, [t.text for t in tokens[i:i + n]])
            if hit:
                return replace(hit, start=tokens[i].start, end=tokens[i + n - 1].end), n
        return None

    def _match(self, attr_id: str, words: list[str]) -> DictMatch | None:
        exact = self._exact.get(attr_id)
        if not exact or not words:
            return None
        phrase = " ".join(words)
        key_fn = self._key_fns.get(attr_id)
        if key_fn:
            key = key_fn(phrase)
            return exact.get(key) if key else None
        hit = exact.get(phrase)
        if hit is None and attr_id in self._stems:
            hit = self._stems[attr_id].get(" ".join(map(stem, words)))
        return hit
