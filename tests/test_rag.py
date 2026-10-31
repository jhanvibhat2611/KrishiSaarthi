"""Synthetic sources and synthetic human reviews; no signed pilot labels used."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from krishisaarthi.schemas import Query
from krishisaarthi.retrieval.bm25 import BM25Retriever
from krishisaarthi.rag import EvidenceSelector, EvidenceSufficiencyChecker, CitationValidator, PromptBuilder, MockGenerator, RAGPipeline
from krishisaarthi.rag.models import Citation, Claim, GeneratedAnswer, SufficiencyReview, ComponentReview

Q=Query(query_id='synthetic-question',text='Which synthetic evidence is available?')

def cite(p):
    return Citation(passage_id=p.passage_id,start_char=p.evidence_span.start_char,
                    end_char=p.evidence_span.end_char,quote=p.text)

def synthetic_review(pack):
    return SufficiencyReview(evidence_fingerprint=pack.fingerprint,reviewer_id='SYNTHETIC-TEST-ONLY',
        reviewed_at='2025-01-01T00:00:00+00:00',all_question_components_covered=True,applicability_checked=True,
        components=[ComponentReview(component='synthetic component '+str(i),support=[cite(p)]) for i,p in enumerate(pack.passages)])

@pytest.fixture
def selector(document,passages):return EvidenceSelector(BM25Retriever(),[document],passages)

def test_selector_provenance_and_metadata(selector,document):
    pack=selector.select(Q,5)
    assert pack.passages
    for p in pack.passages:
        assert p.text==document.canonical_text[p.evidence_span.start_char:p.evidence_span.end_char]
        assert p.page_references and p.document_id==document.document_id
    assert pack.document_metadata[document.document_id]==document.domain_metadata
    assert pack==selector.select(Q,5)

@pytest.mark.parametrize('mode',['unknown','duplicate','wrong_document','wrong_query','wrong_rank','wrong_span'])
def test_invalid_retrieval_rejected(selector,mode):
    search=selector.retriever.search
    def bad(q,k):
        hits=list(search(q,k));h=hits[0]
        if mode=='duplicate':return [h,h]
        field,value={'unknown':('passage_id','unknown'),'wrong_document':('document_id','other'),
                     'wrong_query':('query_id','other'),'wrong_rank':('rank',2),
                     'wrong_span':('evidence_span',h.evidence_span.model_copy(update={'end_char':h.evidence_span.end_char-1}))}[mode]
        return [h.model_copy(update={field:value})]
    selector.retriever.search=bad
    with pytest.raises(ValueError):selector.select(Q,5)

def test_no_review_no_generator_call(selector):
    class Never:
        def generate(self,*args):raise AssertionError('Generator should not run')
    r=RAGPipeline(selector,Never()).run(Q)
    assert r.status=='insufficient_evidence' and r.generated_answer is None
    assert not r.diagnostics['generator_called']

def test_empty_retrieval(selector):
    selector.retriever.search=lambda q,k:[]
    r=RAGPipeline(selector,MockGenerator()).run(Q)
    assert r.status=='insufficient_evidence' and not r.retrieved_passage_ids

@pytest.mark.parametrize('field,value',[
    ('evidence_fingerprint','changed'),('reviewed_at','2025-01-01T00:00:00'),
    ('reviewer_id',' '),('all_question_components_covered',False),
    ('applicability_checked',False),('unresolved_concerns',['missing second component'])])
def test_sufficiency_rejects_uncertainty(selector,field,value):
    pack=selector.select(Q);review=synthetic_review(pack).model_copy(update={field:value})
    assert not EvidenceSufficiencyChecker().assess(pack,review).sufficient_to_attempt

def test_question_change_invalidates_review(selector):
    review=synthetic_review(selector.select(Q))
    assert not EvidenceSufficiencyChecker().assess(selector.select(Q.model_copy(update={'text':'different question'})),review).sufficient_to_attempt

def test_multi_passage_mock_draft_never_certified(selector):
    pack=selector.select(Q);review=synthetic_review(pack)
    assert len(pack.passages)>=2
    r=RAGPipeline(selector,MockGenerator()).run(Q,review=review)
    assert r.status=='review_required' and len(r.claims)==len(pack.passages)
    assert r.factual_verification=='not_performed' and r.sources[0]['page_references']
    assert r.generated_answer and 'Mock' in r.generated_answer
    assert PromptBuilder().build(pack)==PromptBuilder().build(pack)

@pytest.mark.parametrize('mode',['unknown','offset','quote'])
def test_citation_failures(selector,mode):
    pack=selector.select(Q);c=cite(pack.passages[0])
    c=c.model_copy(update={'passage_id':'wrong'} if mode=='unknown' else {'end_char':c.end_char+1} if mode=='offset' else {'quote':'invented'})
    class Bad:
        def generate(self,*args):return GeneratedAnswer(claims=[Claim(text='synthetic claim',citations=[c])])
    r=RAGPipeline(selector,Bad()).run(Q,review=synthetic_review(pack))
    assert r.status=='invalid_output' and r.generated_answer is None and not r.claims

def test_unsupported_claim_with_valid_quote_not_certified(selector):
    pack=selector.select(Q)
    class Unsupported:
        def generate(self,*args):return GeneratedAnswer(claims=[Claim(text='UNSUPPORTED SYNTHETIC CLAIM',citations=[cite(pack.passages[0])])])
    r=RAGPipeline(selector,Unsupported()).run(Q,review=synthetic_review(pack))
    assert r.status=='review_required' and r.factual_verification=='not_performed'
    assert any('UNVERIFIED' in w for w in r.warnings)
    assert r.claims[0].verification_status=='unverified'

@pytest.mark.parametrize('mode',['error','malformed','abstain','missing'])
def test_generator_failures_and_abstention(selector,mode):
    class G:
        def generate(self,*args):
            if mode=='error':raise RuntimeError('private exception details')
            if mode=='malformed':return {'unexpected':'key'}
            return GeneratedAnswer(abstain=mode=='abstain',missing_context=['second part'] if mode=='missing' else [])
    r=RAGPipeline(selector,G()).run(Q,review=synthetic_review(selector.select(Q)))
    assert r.status==('generator_error' if mode in ['error','malformed'] else 'insufficient_evidence')
    assert r.generated_answer is None and 'private exception details' not in r.model_dump_json()

def test_inference_never_reads_gold(selector,monkeypatch):
    original=Path.open
    def guard(self,*args,**kwargs):
        if 'gold' in str(self) or 'annotation' in str(self):raise AssertionError('Forbidden research label access')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    r=RAGPipeline(selector,MockGenerator()).run(Q,review=synthetic_review(selector.select(Q)))
    assert r.status=='review_required'
    import inspect
    import krishisaarthi.rag.models as models
    assert 'GoldLabel' not in inspect.getsource(models)


def test_synthetic_cli_demonstration():
    result=subprocess.run([sys.executable,'scripts/demo_rag.py'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    response=json.loads(result.stdout)
    assert response['status']=='review_required' and response['factual_verification']=='not_performed'
    assert all('synthetic' in s['document_id'] for s in response['sources'])


def test_research_draft_explicit_and_unverified(selector):
    r=RAGPipeline(selector,MockGenerator()).run(Q,mode='research_draft')
    assert r.status=='review_required' and r.answerability=='not_established'
    assert r.execution_mode=='research_draft' and all(c.verification_status=='unverified' for c in r.claims)
    assert any('NOT established' in w for w in r.warnings)
    with pytest.raises(ValueError):RAGPipeline(selector,MockGenerator()).run(Q,mode='unknown')


def test_research_draft_cannot_bypass_empty_or_invalid_sources(selector):
    selector.retriever.search=lambda q,k:[]
    r=RAGPipeline(selector,MockGenerator()).run(Q,mode='research_draft')
    assert r.status=='insufficient_evidence' and not r.diagnostics['generator_called']


def test_research_draft_invalid_citation_not_exposed(selector):
    class Bad:
        def generate(self,*args):
            return GeneratedAnswer(claims=[Claim(text='bad',citations=[Citation(passage_id='absent',start_char=0,end_char=1,quote='x')])])
    r=RAGPipeline(selector,Bad()).run(Q,mode='research_draft')
    assert r.status=='invalid_output' and r.generated_answer is None


def test_research_draft_abstention_and_missing_context(selector):
    class G:
        def generate(self,*args):return GeneratedAnswer(abstain=True,missing_context=['Not supplied in evidence'])
    r=RAGPipeline(selector,G()).run(Q,mode='research_draft')
    assert r.status=='insufficient_evidence' and 'Not supplied in evidence' in r.warnings
