"""Шаг 7 каскада — значение из карточек той же модели (b2): только если в тексте этой карточки значения нет.

У восстановленных и уценённых карточек диагонали и батареи в тексте нет в 100% случаев (EDA, раздел 7), но их
знают карточки той же модели (бренд + модель в каноне) из train. Такое значение не подтверждено текстом этой
карточки: step=model_match, source=other_card, пониженный скор, всегда на ручную проверку и вне основной метрики
(CLAUDE.md, «Правила работы»). Evidence — фрагмент этой карточки, по которому опознана модель.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

from attrex.contracts import BuildContext, Candidate, Card, CategorySchema, Source, Step
from attrex.normalize.dictionary import top_vote
from attrex.normalize.settings import load_settings

_KB_SQL = """
    WITH k AS (SELECT sku_id FROM core_splits WHERE fold IN (SELECT unnest(?))),
    bm AS (
        SELECT g.sku_id,
               max(g.value) FILTER (WHERE g.attr_id = 'brand') AS brand,
               max(g.value) FILTER (WHERE g.attr_id = 'model') AS model
        FROM core_gold g JOIN k USING (sku_id)
        WHERE g.attr_id IN ('brand', 'model') AND g.value IS NOT NULL
        GROUP BY 1
    ),
    known AS (
        SELECT brand, model FROM bm WHERE brand IS NOT NULL AND model IS NOT NULL
        GROUP BY 1, 2 HAVING count(*) >= ?
    )
    SELECT bm.brand, bm.model, g.attr_id, g.value, count(*)
    FROM core_gold g JOIN bm USING (sku_id) JOIN known USING (brand, model)
    WHERE g.value IS NOT NULL AND g.attr_id NOT IN ('brand', 'model')
    GROUP BY 1, 2, 3, 4
"""


@dataclass(frozen=True)
class ModelMatchSettings:
    min_support: int = 2       # сколько карточек модели с этим атрибутом нужно, чтобы взять значение
    min_share: float = 0.6     # и какая доля из них должна согласиться
    score: float = 0.6         # потолок скора: значение не из текста этой карточки
    open_below: float = 0.7    # атрибут берём, если у прошлых шагов нет кандидата со скором не ниже


class ModelKnowledge:
    """(бренд, модель) → атрибут → значения эталона карточек этой модели."""

    def __init__(self, values: dict[tuple[str, str], dict[str, Counter]]):
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    @classmethod
    def from_rows(cls, rows) -> "ModelKnowledge":
        """rows: (бренд, модель, attr_id, значение, сколько карточек)."""
        values: dict[tuple[str, str], dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        for brand, model, attr_id, value, n in rows:
            values[brand, model][attr_id][value] += n
        return cls({k: dict(v) for k, v in values.items()})

    @classmethod
    def from_context(cls, ctx: BuildContext) -> "ModelKnowledge":
        min_support = ModelMatchSettings(**load_settings().model_match).min_support
        return cls.from_rows(ctx.con.execute(_KB_SQL, [list(ctx.knowledge_folds), min_support]).fetchall())

    def get(self, brand: str, model: str, attr_id: str) -> Counter | None:
        return self._values.get((brand, model), {}).get(attr_id)


class ModelMatchExtractor:
    step = Step.MODEL_MATCH

    def __init__(self, knowledge: ModelKnowledge, settings: ModelMatchSettings):
        self.knowledge, self.settings = knowledge, settings

    def extract(self, card: Card, schema: CategorySchema, prior: list[Candidate]) -> list[Candidate]:
        brand, model = _best(prior, "brand"), _best(prior, "model")
        if brand is None or model is None:
            return []
        out = []
        for attr_id in schema.required:
            if attr_id in ("brand", "model") or any(c.score >= self.settings.open_below
                                                    for c in prior if c.attr_id == attr_id):
                continue
            votes = self.knowledge.get(brand.value, model.value, attr_id)
            if not votes:
                continue
            value, n = top_vote(votes)
            total = sum(votes.values())
            share = n / total
            if total >= self.settings.min_support and share >= self.settings.min_share:
                out.append(Candidate(attr_id=attr_id, value=value, step=self.step, source=Source.OTHER_CARD,
                                     score=round(self.settings.score * share, 4),
                                     value_raw=f"{n} из {total} карточек модели {brand.value} {model.value}",
                                     evidence=model.evidence))
        return out


def _best(prior: list[Candidate], attr_id: str) -> Candidate | None:
    cands = [c for c in prior if c.attr_id == attr_id]
    return max(cands, key=lambda c: c.score) if cands else None


def build(ctx: BuildContext) -> ModelMatchExtractor:
    return ModelMatchExtractor(ctx.resource(ModelKnowledge.from_context),
                               ModelMatchSettings(**load_settings().model_match))
