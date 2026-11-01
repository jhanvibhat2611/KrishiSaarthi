"""Synthetic handle contracts; no research text, questions or labels."""
import json
from types import SimpleNamespace
import pytest
from krishisaarthi.rag.handles import handle_table, parse_handle_answer, resolve_answer, OutputFailure
from krishisaarthi.rag import EvidenceSelector, CitationValidator, RAGPipeline, PromptBuilder
from krishisaarthi.retrieval.bm25 import BM25Retriever
from krishisaarthi.schemas import Query
from krishisaarthi.rag.mlx_generator import MLXGenerator

@pytest.fixture
def pack(document,passages):
    return EvidenceSelector(BM25Retriever(),[document],passages).select(Query(query_id='synthetic',text='synthetic',language='en'))

def payload(handles=None):
    return {'answer_status':'draft','answer_text':'Synthetic claim.','claims':[{'text':'Synthetic claim.','evidence_handles':handles or ['E1']}],'uncertainty':[]}

def test_handle_resolution_preserves_exact_source(pack):
    answer=resolve_answer(parse_handle_answer(json.dumps(payload(['E1','E2']))),pack)
    assert not CitationValidator().validate(answer.claims,pack)
    assert handle_table(pack)==handle_table(pack)
    for c,p in zip(answer.claims[0].citations,pack.passages):
        assert c.quote==p.text and c.start_char==p.evidence_span.start_char and c.end_char==p.evidence_span.end_char
    assert all(c.verification_status=='unverified' for c in answer.claims)
    assert 'offsets' in PromptBuilder().build(pack)[0]['content']

@pytest.mark.parametrize('handles',[['E999'],[' E1'],['e1'],['E0'],['E1','E1'],['E01']])
def test_unknown_or_malformed_handles_rejected(pack,handles):
    with pytest.raises(OutputFailure) as e:resolve_answer(parse_handle_answer(json.dumps(payload(handles))),pack)
    assert e.value.category=='invalid_citation'

@pytest.mark.parametrize('raw,category',[
    ('{','invalid_json'),('```json\n{}\n```','invalid_json'),('{}','schema_failure'),
    ('{"answer_status":"insufficient_evidence","answer_text":"","claims":[],"uncertainty":[]}','schema_failure')])
def test_output_categories(raw,category):
    with pytest.raises(OutputFailure) as e:parse_handle_answer(raw)
    assert e.value.category==category

def test_uncited_answer_text_rejected():
    obj=payload();obj['answer_text']='Additional uncited claim'
    with pytest.raises(OutputFailure):parse_handle_answer(json.dumps(obj))

def test_abstention_has_no_claims(pack):
    a=parse_handle_answer(json.dumps({'answer_status':'insufficient_evidence','answer_text':'','claims':[],'uncertainty':['Missing second component']}))
    result=resolve_answer(a,pack)
    assert result.abstain and not result.claims

@pytest.mark.parametrize('category',['invalid_json','schema_failure','invalid_citation','output_length_exhaustion','model_runtime_error'])
def test_pipeline_reports_failure_not_abstention(document,passages,category):
    class Failed:
        def generate(self,*a):raise OutputFailure(category,'synthetic failure')
    s=EvidenceSelector(BM25Retriever(),[document],passages)
    r=RAGPipeline(s,Failed()).run(Query(query_id='synthetic',text='synthetic'),mode='research_draft')
    assert r.status=='generator_error' and r.failure_category==category and not r.generated_answer

def test_length_exhaustion_even_if_json_complete(monkeypatch,pack):
    import sys
    def stream(*a,**kw):yield SimpleNamespace(text=json.dumps(payload()),finish_reason='length',generation_tokens=10)
    monkeypatch.setitem(sys.modules,'mlx_lm',SimpleNamespace(stream_generate=stream))
    monkeypatch.setitem(sys.modules,'mlx_lm.sample_utils',SimpleNamespace(make_sampler=lambda **kw:None))
    g=MLXGenerator.__new__(MLXGenerator);g.model=None;g.max_tokens=10;g.load_seconds=0;g.load_peak_bytes=1
    g.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:[1])
    g.mx=SimpleNamespace(random=SimpleNamespace(seed=lambda s:None),reset_peak_memory=lambda:None,get_peak_memory=lambda:1)
    g.config={'revision':'a'*40,'inference':{'seed':0,'context_limit':100,'prompt_prefill_step_size':256}}
    with pytest.raises(OutputFailure) as e:g.generate([],pack)
    assert e.value.category=='output_length_exhaustion' and g.last_metrics['formatting_repair_attempts']==0
