"""Versioned records. Offsets are Python Unicode code points, [start, end)."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1


class SourceMetadata(Record):
    document_id: str | None = Field(default=None, min_length=1)
    source_url: str | None = None
    language: Literal["en"] = "en"
    domain_metadata: dict[str, Any] = Field(default_factory=lambda: {"crop": "wheat", "country": "India"})


class PageReference(Record):
    page_number: int | None = Field(default=None, ge=1)
    canonical_start: int = Field(ge=0)
    canonical_end: int = Field(ge=0)
    extracted_text_path: str | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.canonical_end < self.canonical_start:
            raise ValueError("Page offsets are reversed")
        return self


class EvidenceSpan(Record):
    document_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_char <= self.start_char:
            raise ValueError("Evidence spans must be nonempty")
        return self


class Document(Record):
    document_id: str = Field(min_length=1)
    content_sha256: str
    canonical_sha256: str
    original_filename: str
    original_file_path: str
    source_url: str | None = None
    media_type: Literal["application/pdf", "text/plain"]
    language: Literal["en"] = "en"
    domain_metadata: dict[str, Any]
    canonical_text: str
    pages: list[PageReference]
    extraction_metadata: dict[str, Any]


class Passage(Record):
    passage_id: str
    document_id: str
    text: str
    evidence_span: EvidenceSpan
    page_references: list[PageReference]
    token_count: int = Field(gt=0)
    chunking_config_id: str
    original_filename: str
    source_url: str | None = None
    domain_metadata: dict[str, Any]


class Query(Record):
    query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: Literal["en", "hi", "hinglish"] | None = None


class RetrievalResult(Record):
    query_id: str
    passage_id: str
    document_id: str
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    evidence_span: EvidenceSpan
    retriever_id: str


class GoldLabel(Record):
    query_id: str = Field(min_length=1)
    # Required: absent labels must never be inferred to mean unanswerable.
    answerable: bool
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    relevant_passage_ids: list[str] = Field(default_factory=list)
    chunking_config_id: str | None = None

    @model_validator(mode="after")
    def valid_labels(self):
        spans, passages = bool(self.evidence_spans), bool(self.relevant_passage_ids)
        if spans and passages:
            raise ValueError("Choose evidence spans OR passage relevance for each query")
        if self.answerable != (spans or passages):
            raise ValueError("Answerable requires positive labels; unanswerable requires none")
        if passages and not self.chunking_config_id:
            raise ValueError("Passage labels require chunking_config_id")
        if len(set(self.relevant_passage_ids)) != len(self.relevant_passage_ids):
            raise ValueError("Duplicate passage labels")
        keys = [s.model_dump_json() for s in self.evidence_spans]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate evidence labels")
        return self
