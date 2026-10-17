"""Текст карточки для поиска значений: свёртка, токены с позициями, грубая основа слова.

`fold` не меняет длину строки, поэтому позиция совпадения в свёрнутом тексте — это позиция в исходном,
и evidence всегда вырезается из текста карточки дословно.
"""
import re
from dataclasses import dataclass
from functools import cached_property

from attrex.contracts import Card, Source

_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
# слово: буквы и цифры, внутри допускаются - . ' + («tp-link», «6.7», «i5-13420h», «s26+»)
_TOKEN = re.compile(r"\w+(?:[-.'+]\w+)*\+?")
# окончания прилагательных и существительных: «черного» → «черн», «полиэстера» → «полиэстер»
_ENDING = re.compile(r"(ыми|ими|ого|его|ому|ему|ая|яя|ое|ее|ые|ие|ый|ий|ой|ых|их|ую|юю|ом|ем|ым|им|а|я|ы|и|у|ю|е|о|ь)$")
_SPACES = re.compile(r"\s+")


def fold(s: str) -> str:
    """Нижний регистр, ё→е, неразрывный пробел → пробел, «6,7» → «6.7». Длина строки не меняется."""
    low = s.lower()
    if len(low) != len(s):  # у редких символов (İ) нижний регистр длиннее — их оставляем как есть
        low = "".join(c.lower() if len(c.lower()) == 1 else c for c in s)
    return _DECIMAL_COMMA.sub(".", low.replace("ё", "е").replace("\xa0", " "))


def squash(s: str) -> str:
    """Свёртка без разницы в пробелах: для проверки, что фрагмент есть в тексте."""
    return _SPACES.sub(" ", fold(s)).strip()


@dataclass(frozen=True)
class Token:
    text: str      # в свёрнутом виде
    start: int
    end: int


def tokenize(folded: str) -> list[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(folded)]


def phrase_key(s: str) -> str:
    """Ключ фразы для словарей: свёрнутые слова через один пробел («Серый / Космос» → «серый космос»)."""
    return " ".join(t.text for t in tokenize(fold(s)))


def stem(word: str) -> str:
    """Основа склоняемого слова: срезаем окончание, если остаётся ≥ 3 букв. Слова с цифрами не трогаем."""
    m = _ENDING.search(word)
    if m and m.start() >= 3 and word.isalpha():
        return word[:m.start()]
    return word


@dataclass(frozen=True)
class TextField:
    """Поле карточки (заголовок или описание): исходный текст, свёртка и токены считаются один раз."""
    source: Source
    text: str

    @cached_property
    def folded(self) -> str:
        return fold(self.text)

    @cached_property
    def tokens(self) -> list[Token]:
        return tokenize(self.folded)

    def fragment(self, start: int, end: int) -> str:
        return self.text[start:end].strip()


@dataclass(frozen=True)
class CardText:
    """Текстовые поля карточки в порядке доверия: заголовок плотнее и точнее описания."""
    card: Card
    title: TextField
    description: TextField

    @classmethod
    def of(cls, card: Card) -> "CardText":
        return cls(card, TextField(Source.TITLE, card.title or ""), TextField(Source.DESCRIPTION, card.description or ""))

    @property
    def fields(self) -> tuple[TextField, TextField]:
        return self.title, self.description
