"""Bounded synthetic QA. No model, research data, labels, or inference imports.

Only explicit fixture questions are supported. This allowlist is a test-double
contract, NOT a general answerability classifier or a research query rule.
"""
from typing import Literal, Protocol
from pydantic import Field, model_validator
from ..rag.models import Model

QUESTIONS = {
    'soil': {
        'en': 'What are the fictional soil sample labels?',
        'hi': 'काल्पनिक मिट्टी के नमूनों के नाम क्या हैं?',
        'hinglish': 'Fictional mitti ke samples ke naam kya hain?',
    },
    'seed': {
        'en': 'What are the fictional seed batch names?',
        'hi': 'काल्पनिक बीज के बैचों के नाम क्या हैं?',
        'hinglish': 'Fictional beej batches ke naam kya hain?',
    },
}
ANSWERS = {
    'soil': {'en': 'The fictional soil sample labels are Cedar and Willow.', 'hi': 'काल्पनिक मिट्टी के नमूनों के नाम Cedar और Willow हैं।', 'hinglish': 'Fictional mitti ke samples ke naam Cedar aur Willow hain.'},
    'seed': {'en': 'The fictional seed batch names are Aster and Fern.', 'hi': 'काल्पनिक बीज के बैचों के नाम Aster और Fern हैं।', 'hinglish': 'Fictional beej batches ke naam Aster aur Fern hain.'},
}
EXPECTED = {
    'soil': ('synthetic-soil-02', 'demo-notebook-soil', 'SYNTHETIC: A fictional researcher labels wheat soil samples Cedar and Willow before recording observations. These invented labels are not soil classes or fertilizer advice.'),
    'seed': ('synthetic-seed-03', 'demo-notebook-seed', 'SYNTHETIC: An invented seed notebook records the appearance of two fictional wheat seed batches, Aster and Fern. It specifies no variety, seed rate or sowing recommendation.'),
}

class Draft(Model):
    status: Literal['draft', 'insufficient_evidence']
    answer: str
    evidence_handles: list[str]
    @model_validator(mode='after')
    def coherent(self):
        if self.status == 'draft' and (not self.answer.strip() or not self.evidence_handles):
            raise ValueError('Draft needs answer and citations')
        if self.status == 'insufficient_evidence' and (self.answer or self.evidence_handles):
            raise ValueError('Abstention cannot contain answer or citations')
        return self

class Source(Model):
    handle: str
    passage_id: str
    document_id: str
    section: str

class Response(Model):
    original_question: str
    language: Literal['en','hi','hinglish']
    query_type: Literal['supported_fixture_lookup','outside_fixture_scope']
    mode: Literal['synthetic_test_double'] = 'synthetic_test_double'
    generator: str = 'deterministic-fixture-v1'
    status: Literal['review_required','UNANSWERABLE','error']
    answerability: Literal['fixture_supported','insufficient_demo_evidence','not_established']
    answer: str | None = None
    citations: list[Source] = Field(default_factory=list)
    confidence: Literal['not_estimated'] = 'not_estimated'
    factual_verification: Literal['not_independently_verified'] = 'not_independently_verified'
    failure_category: str | None = None
    reason: str
    @model_validator(mode='after')
    def coherent(self):
        if self.status == 'review_required':
            if not self.answer or not self.citations or self.answerability != 'fixture_supported' or self.failure_category:
                raise ValueError('Invalid draft response')
        elif self.answer is not None or self.citations:
            raise ValueError('Non-answer cannot include answer or citations')
        if self.status == 'error' and (not self.failure_category or self.answerability != 'not_established'):
            raise ValueError('Failures are not unanswerable judgments')
        return self

class EvidenceOnlyGenerator(Protocol):
    name: str
    def generate(self, *, language: str, fixture: str, evidence: dict[str, dict]) -> dict: ...

class DeterministicGenerator:
    """A finite synthetic test double, not an LLM or general-purpose QA system."""
    name = 'deterministic-fixture-v1'
    def generate(self, *, language, fixture, evidence):
        pid, doc, text = EXPECTED[fixture]
        handles = [h for h,p in evidence.items() if (p.get('passage_id'),p.get('document_id'),p.get('text')) == (pid,doc,text)]
        if not handles:
            return {'status':'insufficient_evidence','answer':'','evidence_handles':[]}
        return {'status':'draft','answer':ANSWERS[fixture][language],'evidence_handles':[handles[0]]}

def answer_question(query, language, passages, generator=None):
    generator = generator or DeterministicGenerator()
    fixture = next((key for key,variants in QUESTIONS.items() if query.strip().casefold() == variants.get(language,'').casefold()), None)
    common = dict(original_question=query,language=language,query_type='supported_fixture_lookup' if fixture else 'outside_fixture_scope',generator=generator.name)
    def abstain(reason):
        return Response(**common,status='UNANSWERABLE',answerability='insufficient_demo_evidence',reason=reason)
    def error(category):
        return Response(**common,status='error',answerability='not_established',failure_category=category,reason='Synthetic QA failed safely. No answerability conclusion was established.')
    if not fixture:
        return abstain('Outside the finite test-double question set. This is a demo abstention, not proof that real evidence is unavailable.')
    evidence={f'E{i}':p for i,p in enumerate(passages,1)}
    pid,doc,text=EXPECTED[fixture]
    if not any((p.get('passage_id'),p.get('document_id'),p.get('text'))==(pid,doc,text) for p in passages):
        return abstain('Required intact synthetic evidence was not retrieved. No unsupported answer is supplied.')
    try:
        raw=generator.generate(language=language,fixture=fixture,evidence=evidence)
    except Exception:
        return error('generator_failure')
    try:
        draft=Draft.model_validate(raw)
    except Exception:
        return error('malformed_output')
    if draft.status=='insufficient_evidence':
        return abstain('The adapter declined to answer from supplied demo evidence; not a verified research label.')
    if len(set(draft.evidence_handles))!=len(draft.evidence_handles) or any(h not in evidence for h in draft.evidence_handles):
        return error('invalid_citation')
    # Synthetic-only content check. Never used with real source passages.
    if draft.answer!=ANSWERS[fixture][language] or any((evidence[h].get('passage_id'),evidence[h].get('document_id'),evidence[h].get('text'))!=(pid,doc,text) for h in draft.evidence_handles):
        return error('unsupported_fixture_output')
    sources=[Source(handle=h,passage_id=evidence[h]['passage_id'],document_id=evidence[h]['document_id'],section=evidence[h]['section']) for h in draft.evidence_handles]
    return Response(**common,status='review_required',answerability='fixture_supported',answer=draft.answer,citations=sources,reason='Deterministic synthetic template, not LLM generation. Citation resolution is structural, not independent factual verification.')
