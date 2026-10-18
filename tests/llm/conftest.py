"""Поддельный транспорт: ответы по очереди, считает вызовы. Сеть и ключи в тестах не нужны."""
import json

import numpy as np
import pytest

from attrex.llm.client import ChatResult


class FakeTransport:
    """errors — исключения для первых вызовов по очереди (None — вызов удаётся); fail_always — каждый вызов падает."""

    def __init__(self, answers=(), prompt_tokens=1000, completion_tokens=100, errors=(), fail_always=None):
        self.answers = list(answers)
        self.errors = list(errors)
        self.fail_always = fail_always
        self.calls = 0
        self.embed_calls = 0
        self.extras = []
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens

    def _maybe_fail(self):
        if self.fail_always is not None:
            raise self.fail_always
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error

    def chat(self, model, messages, temperature, max_tokens, seed, extra):
        self.calls += 1
        self.extras.append(extra)
        self._maybe_fail()
        answer = self.answers[(self.calls - 1) % len(self.answers)] if self.answers else {}
        text = answer if isinstance(answer, str) else "```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```"
        return ChatResult(text, self.prompt_tokens, self.completion_tokens)

    def embed(self, model, texts):
        self.embed_calls += 1
        self._maybe_fail()
        return np.ones((len(texts), 4), dtype=np.float32), 10 * len(texts)


@pytest.fixture
def fake():
    return FakeTransport
