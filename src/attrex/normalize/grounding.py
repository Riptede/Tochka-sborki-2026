"""Grounding: значение подтверждено текстом этой карточки. Проверка одна на все шаги, включая LLM (CLAUDE.md).

Кандидат проходит, если (1) evidence дословно (с точностью до регистра и пробелов) есть в заголовке, описании или
атрибутах продавца карточки и (2) evidence после нормализации содержит значение: «256GB» подтверждает «256»,
«Titanium Grey» — «серый», «Intel Core i5-13420H» — «intel core i5-13420h 2.1 ггц».
"""
from attrex.contracts import Candidate, Card
from attrex.normalize.text import squash
from attrex.normalize.values import ValueNormalizer


def card_texts(card: Card) -> list[str]:
    """Все тексты карточки, где может лежать evidence."""
    return [card.title or "", card.description or "", *(f"{k}: {v}" for k, v in card.seller_attrs.items())]


class Grounding:
    def __init__(self, normalizer: ValueNormalizer):
        self.normalizer = normalizer

    def occurs(self, evidence: str, card: Card) -> bool:
        needle = squash(evidence)
        return bool(needle) and any(needle in squash(t) for t in card_texts(card))

    def check(self, card: Card, cand: Candidate) -> bool:
        return (bool(cand.evidence) and self.occurs(cand.evidence, card)
                and self.normalizer.supports(cand.attr_id, cand.evidence, cand.value))
