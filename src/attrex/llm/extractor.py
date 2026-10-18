"""Шаг 6 каскада — LLM (b3): только пустые и неуверенные атрибуты, где в тексте есть зацепка.

Зацепка — ключевое слово или алиас атрибута в тексте карточки либо кандидат прошлых шагов ниже порога: если
в тексте о значении ничего нет, LLM его не найдёт честно, а только выдумает (EDA, раздел 7).

Промпт: заголовок и описание карточки, атрибуты с типом и единицей, для справочных — варианты: значения
кандидатов прошлых шагов и самые частые значения справочника. Ответ — по JSON-схеме запроса (llm/schema.py):
у справочных атрибутов value — только из этих вариантов, у чисел — число и единица, evidence — дословная цитата.
Значение приводится к канону общей библиотекой b2 (ValueNormalizer) и проходит тот же grounding, что и правила;
без цитаты из этой карточки оно отбрасывается (счётчик Usage.llm_ungrounded).
Self-consistency: несколько ответов с температурой; значение — по большинству, скор — доля голосов.
"""
import re
from collections import Counter
from dataclasses import dataclass

from attrex.confidence import aggregate
from attrex.confidence.aggregate import Aggregator
from attrex.contracts import BuildContext, Candidate, Card, CategorySchema, Source, Step, Usage
from attrex.llm.client import LlmClient
from attrex.llm.schema import AnswerSchema, AttrField
from attrex.normalize.resources import NormalizeResources
from attrex.normalize.text import fold
from attrex.paths import PROCESSED

# что модель добавляет к цитате сверх текста карточки: подпись поля из промпта и выделение markdown
_PROMPT_LABEL = re.compile(r"^\s*(?:заголовок|описание)\s*:\s*", re.IGNORECASE)
_MARKUP = re.compile(r"\*\*|__|`")

SYSTEM = (
    "Ты извлекаешь характеристики товара из текста его карточки. Бери значение только из текста: если его там нет, "
    "верни null. Для каждого значения приведи evidence — точную цитату из текста, где оно указано. "
    "Числа возвращай числом, а единицу измерения — отдельно, как в тексте. Если даны варианты, выбери подходящий "
    "вариант или null. Ответ — только JSON-объект по заданной схеме, без пояснений."
)


@dataclass(frozen=True)
class LlmSettings:
    samples: int = 3
    temperature: float = 0.7
    max_description_chars: int = 2000
    max_options: int = 30
    score: float = 0.85

    @classmethod
    def from_config(cls, llm: dict) -> "LlmSettings":
        return cls(int(llm.get("self_consistency_samples", 3)), float(llm.get("temperature", 0.7)),
                   int(llm.get("max_description_chars", 2000)), int(llm.get("max_options", 30)),
                   float(llm.get("score", 0.85)))


class LlmExtractor:
    step = Step.LLM

    def __init__(self, client: LlmClient, res: NormalizeResources, aggregator: Aggregator, settings: LlmSettings):
        self.client, self.res, self.aggregator, self.settings = client, res, aggregator, settings
        self._hooks = {a: tuple({*res.settings.keywords.get(a, ()), *res.settings.aliases.get(a, ())})
                       for a in res.registry.attrs}
        self._stats = Usage()

    def extract(self, card: Card, schema: CategorySchema, prior: list[Candidate]) -> list[Candidate]:
        if not self.client.available:
            return []
        targets = self.targets(card, schema, prior)
        if not targets:
            return []
        answer_schema = self.answer_schema(targets, prior)
        messages = self.messages(card, targets, prior)
        n = max(1, self.settings.samples)
        temperature = self.settings.temperature if n > 1 else 0.0
        answers = [a for a in (self.client.chat_json(messages, temperature, seed=i, schema=answer_schema)
                               for i in range(n)) if a]
        self._stats = self._stats + Usage(llm_cards=1)
        return self._vote(card, answer_schema, answers, n)

    def drain_usage(self) -> Usage:
        stats, self._stats = self._stats, Usage()
        return self.client.drain_usage() + stats

    def targets(self, card: Card, schema: CategorySchema, prior: list[Candidate]) -> list[str]:
        """Пустые и ниже порога атрибуты схемы, у которых в тексте есть зацепка."""
        decisions = self.aggregator.decide(prior, schema.required)
        text = fold(f"{card.title}\n{card.description}")
        out = []
        for a in schema.required:
            d = decisions[a]
            if not d.empty and d.confidence >= self.aggregator.threshold(a):
                continue
            if not d.empty or any(h in text for h in self._hooks.get(a, ())):
                out.append(a)
        return out

    def answer_schema(self, targets: list[str], prior: list[Candidate]) -> AnswerSchema:
        """Схема ответа: справочный атрибут с вариантами — enum, число — число и единица, остальное — строка."""
        fields = []
        for a in targets:
            options = self._options(a, prior)
            kind = "number" if self.res.registry[a].kind == "number" else "enum" if options else "text"
            fields.append(AttrField(a, kind, tuple(options)))
        return AnswerSchema(fields)

    def messages(self, card: Card, targets: list[str], prior: list[Candidate]) -> list[dict]:
        lines = []
        for a in targets:
            spec = self.res.registry[a]
            what = f"{a} — {spec.name_ru}" + (f", число в {spec.unit}" if spec.unit else "")
            options = self._options(a, prior)
            lines.append(what + (f"; варианты: {', '.join(options)}" if options else ""))
        description = (card.description or "")[: self.settings.max_description_chars]
        user = (f"Заголовок: {card.title}\nОписание: {description}\n\nХарактеристики:\n- " + "\n- ".join(lines)
                + "\n\nДля каждой характеристики верни value и evidence (у чисел ещё unit). "
                  "Если значения в тексте нет — value и evidence null.")
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def _options(self, attr_id: str, prior: list[Candidate]) -> list[str]:
        if self.res.registry[attr_id].kind != "category" or attr_id == "brand":
            return []
        seen = [c.value for c in prior if c.attr_id == attr_id]
        support = self.res.dictionary.value_support(attr_id)
        # при равной частоте — по алфавиту: промпт и ключ кэша не должны зависеть от порядка строк из БД
        frequent = sorted(support, key=lambda v: (-support[v], v))
        out: list[str] = []
        for v in [*seen, *frequent]:
            if v not in out:
                out.append(v)
            if len(out) >= self.settings.max_options:
                break
        return out

    def _vote(self, card: Card, schema: AnswerSchema, answers: list[dict], n: int) -> list[Candidate]:
        out = []
        for a, field in schema.fields.items():
            votes: Counter = Counter()
            proof: dict[str, Candidate] = {}
            for ans in answers:
                cand = self._candidate(card, field, ans[a])
                if cand:
                    votes[cand.value] += 1
                    proof.setdefault(cand.value, cand)
            if votes:
                value, k = votes.most_common(1)[0]
                c = proof[value]
                out.append(Candidate(a, value, self.step, c.source, round(self.settings.score * k / n, 4),
                                     c.value_raw, c.evidence))
        return out

    def _candidate(self, card: Card, field: AttrField, item: dict) -> Candidate | None:
        """Ответ по атрибуту (уже по схеме) → кандидат в каноне, если цитата есть в карточке и подтверждает значение.

        Непустое значение без цитаты, не приводимое к канону или не прошедшее grounding — счётчик llm_ungrounded.
        """
        raw = field.raw_value(item)
        if raw is None:
            return None
        self._stats = self._stats + Usage(llm_values=1)
        evidence = clean_evidence(item.get("evidence") or "")
        canon = self.res.normalizer.canonical(field.attr_id, raw) if evidence else None
        cand = None
        if canon is not None:
            source = Source.TITLE if fold(evidence) in fold(card.title) else Source.DESCRIPTION
            cand = Candidate(field.attr_id, canon.value, self.step, source, canon.score, raw, evidence)
        if cand is None or not self.res.grounding.check(card, cand):
            self._stats = self._stats + Usage(llm_ungrounded=1)
            return None
        return cand


def clean_evidence(text: str) -> str:
    """Цитата без того, что добавила модель: «Заголовок: …» из промпта и **выделение**. Остальное — как в ответе:
    дальше grounding проверяет, что цитата есть в карточке и подтверждает значение."""
    return _PROMPT_LABEL.sub("", _MARKUP.sub("", text)).strip()


def build(ctx: BuildContext) -> LlmExtractor:
    llm, embeddings = ctx.config.get("llm", {}), ctx.config.get("embeddings", {})
    client = LlmClient.from_config(llm, embeddings, PROCESSED / llm.get("cache", "llm_cache.sqlite"))
    return LlmExtractor(client, ctx.resource(NormalizeResources.from_context), ctx.resource(aggregate.build),
                        LlmSettings.from_config(llm))
