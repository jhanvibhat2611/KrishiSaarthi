"""Baseline tokens only. Future model adapters must enforce their own limits."""
from dataclasses import dataclass
from typing import Protocol, Sequence
import regex

TOKENIZER_ID = "unicode-words-marks-numbers-punctuation-v1"
_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    text: str


class SpanTokenizer(Protocol):
    def token_spans(self, text: str) -> Sequence[TokenSpan]: ...


class UnicodeTokenizer:
    def token_spans(self, text: str) -> list[TokenSpan]:
        return [TokenSpan(m.start(), m.end(), m.group()) for m in _PATTERN.finditer(text)]

    def terms(self, text: str) -> list[str]:
        return [token.text.casefold() for token in self.token_spans(text)]
