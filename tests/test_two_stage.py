"""Synthetic two-stage controls; no gold labels or actual model required."""
import json
from types import SimpleNamespace
import pytest
from krishisaarthi.rag.two_stage import TwoStageGenerator, repeating_tail
from krishisaarthi.rag.handles import OutputFailure
from krishisaarthi.rag import RAGPipeline,EvidenceSelector,CitationValidator
from krishisaarthi.retrieval.bm25 import BM25Retriever
from krishisaarthi.schemas import Query

class Backend:
    def __init__(self,outputs):self.outputs=iter(outputs);self.calls=[];self.last_metrics={}
    def decode(self,messages,max_tokens):
        self.calls.append((messages,max_tokens));self.last_output_text=next(self.outputs)
        self.last_metrics={'output_tokens':3,'mlx_peak_bytes':1,'runtime_seconds':.01}
        return self.last_output_text

@pytest.fixture
def selection(document,passages):return EvidenceSelector(BM25Retriever(),[document],passages)
Q=Query(query_id='synthetic',text='Compare the synthetic descriptions.',language='hi')
A=json.dumps({'status':'candidate','handles':['E1','E3'],'reason':''})
B=json.dumps({'status':'draft','claims':[{'text':'Synthetic claim.','evidence_handles':['E3']}],'reason':''})

def test_candidate_is_not_certification_and_b_sees_only_selected(selection):
    backend=Backend([A,B]);g=TwoStageGenerator(backend);r=RAGPipeline(selection,g).run(Q,mode='research_draft')
    assert r.status=='review_required' and r.factual_verification=='not_performed'
    sources=json.loads(backend.calls[1][0][1]['content'])['sources']
    assert [p['handle'] for p in sources]==['E1','E3']
    assert 'answer_text' not in json.loads(B)
    assert [c[1] for c in backend.calls]==[192,384]
    assert not CitationValidator().validate(r.claims,selection.select(Q))
    assert not g.last_metrics['candidate_selection_is_verification']

def test_default_still_does_not_generate(selection):
    b=Backend([]);r=RAGPipeline(selection,TwoStageGenerator(b)).run(Q)
    assert r.status=='insufficient_evidence' and not b.calls

@pytest.mark.parametrize('status',['insufficient','uncertain'])
def test_a_abstention_skips_b(selection,status):
    b=Backend([json.dumps({'status':status,'handles':[],'reason':'Missing synthetic component'})])
    r=RAGPipeline(selection,TwoStageGenerator(b)).run(Q,mode='research_draft')
    assert r.status=='insufficient_evidence' and len(b.calls)==1
    assert any(status+':' in w for w in r.warnings)

def test_b_can_override_candidate(selection):
    b=Backend([A,json.dumps({'status':'uncertain','claims':[],'reason':'Source ambiguous'})])
    r=RAGPipeline(selection,TwoStageGenerator(b)).run(Q,mode='research_draft')
    assert r.status=='insufficient_evidence' and not r.claims

@pytest.mark.parametrize('raw,category',[
 ('{','invalid_json'),('{}','schema_failure'),
 (json.dumps({'status':'candidate','handles':[],'reason':''}),'incorrect_status'),
 (json.dumps({'status':'insufficient','handles':['E1'],'reason':'missing'}),'incorrect_status'),
 (json.dumps({'status':'candidate','handles':['E99'],'reason':''}),'invalid_citation')])
def test_stage_a_failure_categories(selection,raw,category):
    r=RAGPipeline(selection,TwoStageGenerator(Backend([raw]))).run(Q,mode='research_draft')
    assert r.status=='generator_error' and r.failure_category==category

@pytest.mark.parametrize('handle',['E2','E99',' E1'])
def test_stage_b_cannot_cite_unselected_handle(selection,handle):
    b=json.dumps({'status':'draft','claims':[{'text':'Synthetic','evidence_handles':[handle]}],'reason':''})
    r=RAGPipeline(selection,TwoStageGenerator(Backend([A,b]))).run(Q,mode='research_draft')
    assert r.failure_category=='invalid_citation' and not r.claims

def test_repeat_guard_and_unsupported_constrained_mode():
    assert repeating_tail('Hindi word '*6)
    assert not repeating_tail('Different short synthetic evidence text.')
    with pytest.raises(NotImplementedError):TwoStageGenerator(Backend([]),constrained_json=True)

@pytest.mark.parametrize('text,finish,category',[
    ('{"status":"uncertain"}','stop',None),
    ('partial','length','output_length_exhaustion'),
    ('repeat word '*6,None,'repetition')])
def test_raw_backend_stops_and_records(monkeypatch,text,finish,category):
    import sys
    from krishisaarthi.rag.two_stage import MLXTextBackend
    closed=[]
    def stream(*args,**kwargs):
        try:yield SimpleNamespace(text=text,finish_reason=finish,generation_tokens=7)
        finally:closed.append(True)
    monkeypatch.setitem(sys.modules,'mlx_lm',SimpleNamespace(stream_generate=stream))
    monkeypatch.setitem(sys.modules,'mlx_lm.sample_utils',SimpleNamespace(make_sampler=lambda **kw:None))
    g=MLXTextBackend.__new__(MLXTextBackend);g.model=None
    g.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:[1,2])
    g.mx=SimpleNamespace(random=SimpleNamespace(seed=lambda s:None),reset_peak_memory=lambda:None,get_peak_memory=lambda:100)
    g.config={'inference':{'seed':0,'context_limit':100,'prompt_prefill_step_size':256}}
    if category:
        with pytest.raises(OutputFailure) as e:g.decode([],20)
        assert e.value.category==category
    else:assert g.decode([],20)==text
    assert closed and g.last_output_text==text and g.last_metrics['output_tokens']==7

def test_candidate_reason_is_retained_but_not_used_as_evidence(selection):
    a=json.loads(A);a['reason']='UNVERIFIED diagnostic, not source evidence'
    b=json.loads(B);b['reason']='Unverified model explanation'
    backend=Backend([json.dumps(a),json.dumps(b)]);g=TwoStageGenerator(backend)
    r=RAGPipeline(selection,g).run(Q,mode='research_draft')
    assert r.status=='review_required'
    assert g.last_stages[0]['parsed']['reason']==a['reason']
    assert a['reason'] not in json.dumps(backend.calls[1][0])
    assert b['reason'] not in r.generated_answer
