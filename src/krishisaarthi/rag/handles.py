"""Request-local handles bind model choices to application-owned source spans."""
import json
from pydantic import Field, ValidationError, model_validator
from typing import Literal
from .models import Model, Citation, Claim, GeneratedAnswer
from ..validation import require

class OutputFailure(ValueError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category

class HandleClaim(Model):
    text: str = Field(min_length=1)
    evidence_handles: list[str] = Field(min_length=1)

class HandleAnswer(Model):
    answer_status: Literal['draft', 'insufficient_evidence']
    answer_text: str
    claims: list[HandleClaim]
    uncertainty: list[str]

    @model_validator(mode='after')
    def consistent_status(self):
        if self.answer_status == 'insufficient_evidence':
            if self.answer_text or self.claims or not any(s.strip() for s in self.uncertainty):
                raise ValueError('Abstention requires empty answer/claims and explicit uncertainty')
        elif not self.answer_text.strip() or not self.claims:
            raise ValueError('Draft requires answer text and claims')
        # No additional uncited prose may be introduced through answer_text.
        elif self.answer_text != '\n'.join(c.text for c in self.claims):
            raise ValueError('answer_text must equal claim texts joined with newline')
        return self

def handle_table(evidence):
    return {f'E{i}': p for i, p in enumerate(evidence.passages, 1)}

def parse_handle_answer(raw):
    try:
        json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise OutputFailure('invalid_json', 'Model output is not a JSON object') from exc
    try:
        return HandleAnswer.model_validate_json(raw, strict=True)
    except ValidationError as exc:
        raise OutputFailure('schema_failure', 'Model output violates the handle schema') from exc

def resolve_answer(answer, evidence):
    table = handle_table(evidence)
    claims = []
    for claim in answer.claims:
        citations = []
        for handle in claim.evidence_handles:
            # Exact lookup: no trimming, fuzzy matching, remapping or guessed handles.
            if handle not in table or claim.evidence_handles.count(handle) != 1:
                raise OutputFailure('invalid_citation', 'Unknown, malformed or duplicate evidence handle')
            p = table[handle]
            require(len(p.text) == p.evidence_span.end_char-p.evidence_span.start_char,
                    'Source span length mismatch')
            citations.append(Citation(passage_id=p.passage_id, start_char=p.evidence_span.start_char,
                end_char=p.evidence_span.end_char, quote=p.text))
        claims.append(Claim(text=claim.text, citations=citations))
    return GeneratedAnswer(claims=claims, abstain=answer.answer_status=='insufficient_evidence',
                           missing_context=answer.uncertainty)
