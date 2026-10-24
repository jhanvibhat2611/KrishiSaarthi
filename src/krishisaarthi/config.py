import hashlib
import json
from pathlib import Path
import tomllib
import regex
from pydantic import Field, model_validator
from .schemas import Record
from typing import Literal


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkingConfig(Record):
    size: int = Field(default=256, gt=0)
    overlap: int = Field(default=48, ge=0)
    heading_pattern: str = r"^#{1,6}[ \t]+[^\n]+$"

    @model_validator(mode="after")
    def valid(self):
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        regex.compile(self.heading_pattern, regex.MULTILINE)
        return self


class BM25Config(Record):
    k1: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0, le=1)
    epsilon: float = Field(default=0.25, ge=0, allow_inf_nan=False)


class EvaluationConfig(Record):
    ks: list[int] = Field(default_factory=lambda: [1, 5, 10], min_length=1)
    coverage_rule: Literal["strict_union"] = "strict_union"

    @model_validator(mode="after")
    def valid(self):
        if any(k <= 0 for k in self.ks) or len(set(self.ks)) != len(self.ks):
            raise ValueError("K values must be distinct positive integers")
        return self


class Config(Record):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(path: Path) -> Config:
    with path.open("rb") as stream:
        return Config.model_validate(tomllib.load(stream))
