"""RAG-only contracts. No evaluation labels or inference-time gold records."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from ..schemas import Passage, Query
from ..config import fingerprint

class Model(BaseModel):
    model_config = ConfigDict(extra='forbid')

class EvidencePack(Model):
    question: Query
    passages: list[Passage]
    retriever_id: str
    document_metadata: dict = Field(default_factory=dict)
    @property
    def fingerprint(self):
        return fingerprint(self.model_dump())

class Citation(Model):
    passage_id: str
    start_char: int
    end_char: int
    quote: str = Field(min_length=1)

class Claim(Model):
    verification_status: Literal['unverified'] = 'unverified'
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)

class GeneratedAnswer(Model):
    claims: list[Claim] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    abstain: bool = False

class ComponentReview(Model):
    component: str = Field(min_length=1)
    support: list[Citation] = Field(min_length=1)

class SufficiencyReview(Model):
    """Explicit per-request human support review, never an annotation conversion."""
    evidence_fingerprint: str
    reviewer_id: str = Field(min_length=1)
    reviewed_at: str
    all_question_components_covered: bool
    applicability_checked: bool
    components: list[ComponentReview] = Field(min_length=1)
    unresolved_concerns: list[str] = Field(default_factory=list)

class Sufficiency(Model):
    sufficient_to_attempt: bool
    reasons: list[str]

class RAGResponse(Model):
    execution_mode: Literal['conservative','research_draft'] = 'conservative'
    original_question: str
    retrieved_passage_ids: list[str]
    evidence_fingerprint: str
    status: Literal['insufficient_evidence','review_required','invalid_output','generator_error']
    answerability: Literal['not_established','sufficient_to_attempt']
    generated_answer: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    factual_verification: Literal['not_performed'] = 'not_performed'
    diagnostics: dict = Field(default_factory=dict)
