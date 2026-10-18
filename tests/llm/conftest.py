"""Поддельный транспорт: ответы по очереди, считает вызовы. Сеть и ключи в тестах не нужны."""
import json

import numpy as np
import pytest

from attrex.llm.client import ChatResult


class FakeTransport:
    def __init__(self, answers=(), prompt_tokens=1000, completion_tokens=100):
        self.answers = list(answers)
        self.calls = 0
        self.embed_calls = 0
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens

    def chat(self, model, messages, temperature, max_tokens, seed, extra):
        answer = self.answers[self.calls % len(self.answers)] if self.answers else {}
        self.calls += 1
        text = answer if isinstance(answer, str) else "```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```"
        return ChatResult(text, self.prompt_tokens, self.completion_tokens)

    def embed(self, model, texts):
        self.embed_calls += 1
        return np.ones((len(texts), 4), dtype=np.float32), 10 * len(texts)


@pytest.fixture
def fake():
    return FakeTransport
