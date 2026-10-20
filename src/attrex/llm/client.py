"""Клиент LLM и эмбеддингов (b3) — единственная точка платных вызовов проекта, в том числе для retrieval (b2).

- Адрес и ключ — из окружения ВМ (OPENAI_BASE_URL, OPENAI_API_KEY): их читает клиент openai, в код и логи ключ
  не попадает. Без ключа (например, на ноутбуке) client.available == False: шаги на нём выключаются, а не падают.
- Кэш на диске (sqlite): одинаковый запрос второй раз не оплачиваем, а повторный прогон воспроизводим.
- Бюджет budget_rub: до вызова оцениваем стоимость по худшему случаю (max_tokens ответа); не влезает — вызова нет.
  Бюджет считается на процесс: LLM-прогон — с --workers 1.
- Сбои: таймаут (llm.timeout_s) и повторы (llm.max_retries) — из конфига. Любая ошибка API после всех повторов даёт
  None и счётчик Usage.api_failures, прогон не падает. После MAX_CONSECUTIVE_FAILURES сбоев подряд клиент больше
  не зовёт API: иначе таймауты растянули бы прогон на часы, а ответов всё равно не будет.
- Ответ чата проверяется JSON-схемой запроса (llm/schema.py): не по схеме — None и счётчик Usage.llm_invalid.
- Учёт: вызовы, токены и рубли (цены — configs/pipeline.yaml) отдаёт drain_usage(), pipeline пишет их в pred_runs.
"""
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from attrex.contracts import Usage
from attrex.llm.schema import AnswerSchema

CHARS_PER_TOKEN = 3          # грубая оценка токенов промпта до вызова: для русского текста с запасом
MAX_CONSECUTIVE_FAILURES = 10
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    pass


class PermanentApiError(RuntimeError):
    """Ошибка, которую повтор не исправит (неверный запрос, нет доступа): повторов не делаем."""


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class Transport(Protocol):
    """Сетевая часть: в тестах — подделка, на ВМ — OpenAiTransport."""

    def chat(self, model: str, messages: list[dict], temperature: float, max_tokens: int, seed: int | None,
             extra: dict) -> ChatResult: ...

    def embed(self, model: str, texts: Sequence[str]) -> tuple[np.ndarray, int]: ...


class OpenAiTransport:
    """OpenAI-совместимый API платформы; openai — опциональная зависимость (uv sync --extra llm).

    Повторы делает LlmClient (одинаково для любого транспорта), поэтому у клиента openai они выключены.
    """

    def __init__(self, timeout: float = 60.0):
        import openai  # здесь, а не в начале модуля: без extra llm пакета нет, а остальной проект работает

        self._client = openai.OpenAI(timeout=timeout, max_retries=0)
        self._permanent = (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError,
                           openai.NotFoundError, openai.UnprocessableEntityError)

    def chat(self, model, messages, temperature, max_tokens, seed, extra) -> ChatResult:
        try:
            r = self._client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                     max_tokens=max_tokens, seed=seed, **extra)
        except self._permanent as e:
            raise PermanentApiError(f"{type(e).__name__} {e.status_code}") from None   # текст ошибки не логируем
        u = r.usage
        return ChatResult(r.choices[0].message.content or "", u.prompt_tokens if u else 0,
                          u.completion_tokens if u else 0)

    def embed(self, model, texts) -> tuple[np.ndarray, int]:
        try:
            r = self._client.embeddings.create(model=model, input=list(texts))
        except self._permanent as e:
            raise PermanentApiError(f"{type(e).__name__} {e.status_code}") from None
        vectors = np.asarray([d.embedding for d in r.data], dtype=np.float32)
        return vectors, (r.usage.total_tokens if r.usage else 0)


class DiskCache:
    """Ключ → ответ в sqlite. Несколько процессов могут читать и писать: sqlite сам держит блокировки."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str) -> str | None:
        with self._connect() as con:
            row = con.execute("SELECT value FROM cache WHERE key = ?", [key]).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self._connect() as con:
            con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", [key, value])


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def parse_json(text: str) -> dict | None:
    """Первый JSON-объект в ответе модели (модели любят обрамлять его текстом и ```json)."""
    m = _JSON_BLOCK.search(text or "")
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


class LlmClient:
    def __init__(self, model: str, price: tuple[float, float], budget_rub: float, cache: DiskCache | None = None,
                 transport: Transport | None = None, max_tokens: int = 600, extra: dict | None = None,
                 embed_model: str | None = None, embed_price: float = 0.0, embed_budget_rub: float = 0.0,
                 max_retries: int = 2, retry_backoff_s: float = 1.0):
        self.model, self.price, self.budget_rub = model, price, budget_rub
        self.cache, self.transport, self.max_tokens = cache, transport, max_tokens
        self.extra = dict(extra or {})
        self.embed_model, self.embed_price, self.embed_budget_rub = embed_model, embed_price, embed_budget_rub
        self.max_retries, self.retry_backoff_s = max_retries, retry_backoff_s
        self._usage = Usage()
        self.spent_rub = 0.0
        self.embed_spent_rub = 0.0
        self.skipped_by_budget = 0
        self.consecutive_failures = 0

    @property
    def available(self) -> bool:
        return self.transport is not None and self.budget_rub > 0

    @property
    def broken(self) -> bool:
        return self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price[0] + completion_tokens * self.price[1]) / 1000

    def _count(self, **usage) -> None:
        self._usage = self._usage + Usage(**usage)

    def _call(self, fn: Callable, *args):
        """Вызов API с повторами. Любая ошибка после всех попыток — None и счётчик сбоев, а не исключение."""
        if self.broken:
            self._count(llm_skipped=1)
            return None
        for attempt in range(self.max_retries + 1):
            try:
                result = fn(*args)
            except Exception as e:  # noqa: BLE001 — сеть, таймаут, 4xx/5xx: прогон не должен падать
                error = e
                if isinstance(e, PermanentApiError) or attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_s * 2 ** attempt)
            else:
                self.consecutive_failures = 0
                return result
        self.consecutive_failures += 1
        self._count(api_failures=1)
        # только тип ошибки: текст исключения сетевой библиотеки может содержать подробности запроса
        log.warning("API %s: %s, попыток %d", self.model,
                    error if isinstance(error, PermanentApiError) else type(error).__name__, attempt + 1)
        if self.broken:
            log.warning("API: %d сбоев подряд — больше не вызываем до конца процесса", self.consecutive_failures)
        return None

    def chat_json(self, messages: list[dict], temperature: float = 0.0, seed: int | None = None,
                  schema: AnswerSchema | None = None) -> dict | None:
        """Ответ модели как JSON-объект; с schema — по JSON-схеме (она же уходит в запрос как response_format).

        None — нет ключа, кончился бюджет, сбой API или ответ не по схеме (не JSON).
        """
        extra = {**self.extra, "response_format": schema.response_format()} if schema else self.extra
        key = _key("chat", self.model, messages, temperature, seed, self.max_tokens, extra)
        text = self.cache.get(key) if self.cache else None
        if text is None:
            text = self._ask(key, messages, temperature, seed, extra)
            if text is None:
                return None
        answer = schema.parse(text) if schema else parse_json(text)
        if answer is None:
            self._count(llm_invalid=1)
        return answer

    def _ask(self, key: str, messages: list[dict], temperature: float, seed: int | None, extra: dict) -> str | None:
        if not self.available:
            return None
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        worst = self._cost(prompt_chars // CHARS_PER_TOKEN + 1, self.max_tokens)
        if self.spent_rub + worst > self.budget_rub:
            self.skipped_by_budget += 1
            self._count(llm_skipped=1)
            return None
        r = self._call(self.transport.chat, self.model, messages, temperature, self.max_tokens, seed, extra)
        if r is None:
            return None
        cost = self._cost(r.prompt_tokens, r.completion_tokens)
        self.spent_rub += cost
        self._count(llm_calls=1, llm_tokens=r.prompt_tokens + r.completion_tokens, llm_cost_rub=cost)
        if self.cache:
            self.cache.put(key, r.text)
        return r.text

    def embed(self, texts: Sequence[str]) -> np.ndarray | None:
        """L2-нормированные эмбеддинги; уже посчитанные берутся из кэша. None — нет ключа, бюджета или сбой API."""
        out: list[np.ndarray | None] = [None] * len(texts)
        missing = []
        for i, t in enumerate(texts):
            cached = self.cache.get(_key("embed", self.embed_model, t)) if self.cache else None
            if cached is not None:
                out[i] = np.asarray(json.loads(cached), dtype=np.float32)
            else:
                missing.append(i)
        if missing:
            if self.transport is None or not self.embed_model or self.embed_budget_rub <= 0:
                return None
            worst = sum(len(texts[i]) for i in missing) / CHARS_PER_TOKEN * self.embed_price / 1000
            if self.embed_spent_rub + worst > self.embed_budget_rub:
                self.skipped_by_budget += 1
                self._count(llm_skipped=1)
                return None
            r = self._call(self.transport.embed, self.embed_model, [texts[i] for i in missing])
            if r is None:
                return None
            vectors, tokens = r
            cost = tokens * self.embed_price / 1000
            self.embed_spent_rub += cost
            self._count(embed_tokens=tokens, embed_cost_rub=cost)
            for i, v in zip(missing, vectors):
                out[i] = v
                if self.cache:
                    self.cache.put(_key("embed", self.embed_model, texts[i]), json.dumps([round(float(x), 6) for x in v]))
        m = np.stack(out)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def drain_usage(self) -> Usage:
        usage, self._usage = self._usage, Usage()
        return usage

    @property
    def can_embed(self) -> bool:
        return self.transport is not None and bool(self.embed_model) and self.embed_budget_rub > 0

    @classmethod
    def from_config(cls, llm: dict, embeddings: dict, cache_path: Path | None, transport: Transport | None = None
                    ) -> "LlmClient":
        """Клиент по configs/pipeline.yaml. Транспорт создаётся, только если есть бюджет и ключ в окружении."""
        wants = (llm.get("enabled") and llm.get("budget_rub", 0) > 0) or embeddings.get("budget_rub", 0) > 0
        if transport is None and wants and os.environ.get("OPENAI_API_KEY"):
            transport = OpenAiTransport(float(llm.get("timeout_s", 60)))
        model = llm["model"]
        prices = llm.get("prices_rub_per_1k", {})
        if model not in prices:
            raise ValueError(f"нет цены модели {model} в llm.prices_rub_per_1k (configs/pipeline.yaml)")
        extra = {"reasoning_effort": llm["reasoning_effort"]} if llm.get("reasoning_effort") else {}
        embed_model = embeddings.get("model")
        return cls(model, tuple(prices[model]), float(llm.get("budget_rub", 0)),
                   DiskCache(cache_path) if cache_path else None, transport, int(llm.get("max_tokens", 600)), extra,
                   embed_model, float(embeddings.get("prices_rub_per_1k", {}).get(embed_model, 0.0)),
                   float(embeddings.get("budget_rub", 0)), int(llm.get("max_retries", 2)))


class ApiEncoder:
    """Encoder для retrieval (b2) на эмбеддингах платформы: вызовы — через LlmClient (кэш, бюджет, учёт)."""

    def __init__(self, client: LlmClient):
        self.client = client

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.client.embed(texts)
        if vectors is None:
            raise BudgetExceeded("эмбеддинги недоступны: нет ключа, кончился embeddings.budget_rub или API не отвечает")
        return vectors

    def drain_usage(self) -> Usage:
        return self.client.drain_usage()
