"""All decoder/retrieval calls are mocked; sources are existing invented fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from krishisaarthi.evaluation.qa_readiness import E5ComparisonProposal
from krishisaarthi.evaluation.qa_comparison_fake import Question
from krishisaarthi.evaluation.qa_adapters import PairedComparison, E5EvidenceAdapter, QwenDecoder, save_private
from krishisaarthi.rag.models import EvidencePack
from krishisaarthi.schemas import Query
from krishisaarthi.rag.handles import OutputFailure


def proposal():
    return E5ComparisonProposal.model_validate_json(Path('configs/qa-readiness-e5.proposed.json').read_text(),strict=True)


class MockDecoder:
    execution_kind='mock'
    last_output_text=''
    last_metrics={}
    def __init__(self, outputs=None):
        self.calls=[];self.outputs=outputs or [json.dumps({'status':'uncertain','claims':[],'reason':'Synthetic uncertainty'})]*2
    def prompt_tokens(self,messages): return list(range(120 if 'sources' in messages[1]['content'] else 80))
    def decode(self,messages,max_tokens):
        self.last_metrics={};self.last_output_text=''
        self.calls.append((messages,max_tokens))
        value=self.outputs[len(self.calls)-1]
        self.last_metrics={'input_tokens':len(self.prompt_tokens(messages)),'output_tokens':12,'finish_reason':'stop','runtime_seconds':0.1}
        if isinstance(value,Exception):
            self.last_output_text='partial synthetic output';raise value
        self.last_output_text=value;return value


def evidence(document,passages,language='en'):
    return EvidencePack(question=Query(query_id='synthetic',text='Which fictional observations?',language=language),
        passages=passages,retriever_id='synthetic',document_metadata={document.document_id:document.domain_metadata})


def run(document,passages,decoder=None):
    e=evidence(document,passages);runner=PairedComparison(decoder or MockDecoder(),proposal())
    result=runner.pair('pilot-v01-in-06-en',Question(text=e.question.text,language='en'),e,[document])
    return runner,result


def test_isolation_same_question_and_fixed_budget(document,passages):
    d=MockDecoder();r,out=run(document,passages,d)
    assert len(d.calls)==2 and all(cap==384 for _,cap in d.calls)
    direct,rag=[json.loads(c[0][1]['content']) for c in d.calls]
    assert direct==rag['question'] and set(direct)=={'text','language'}
    assert passages[0].text not in json.dumps(d.calls[0][0])
    assert rag['sources'][0]['text']==passages[0].text
    assert out['model_attempts']==0 and out['overall']['attempted']==2 and out['overall']['planned']==24
    assert [x['input_tokens'] for x in out['rows'][:2]]==[80,120]
    assert out['overall']['withholding']==2
    assert all(x['human_review']['approval'] is None for x in out['rows'])
    with pytest.raises(ValueError):r.pair('pilot-v01-in-06-en',Question(text='q',language='en'),evidence(document,passages),[document])


def test_canonical_citations_not_synthetic_zero_offsets(document,passages):
    p=passages[-1];e=evidence(document,[p])
    raw=json.dumps({'status':'draft','claims':[{'text':'Invented fixture claim','evidence_handles':['E1']}],'reason':''})
    d=MockDecoder([json.dumps({'status':'draft','claims':[{'text':'Closed book fixture'}],'reason':''}),raw])
    runner=PairedComparison(d,proposal());out=runner.pair('pilot-v01-in-06-en',Question(text=e.question.text,language='en'),e,[document])
    row=out['rows'][1];c=row['citations'][0]
    assert row['status']=='review_required' and row['factual_support']=='not_verified'
    assert c['start_char']==p.evidence_span.start_char and c['end_char']==p.evidence_span.end_char
    assert c['quote']==p.text and c['document_id']==document.document_id
    assert c['evidence_span']['content_sha256']==document.content_sha256


@pytest.mark.parametrize('handle',['E99',' E1','e1'])
def test_invalid_citations_fail(document,passages,handle):
    raw=json.dumps({'status':'draft','claims':[{'text':'Synthetic','evidence_handles':[handle]}],'reason':''})
    _,r=run(document,passages,MockDecoder(['{bad',raw]))
    assert r['rows'][0]['failure_category']=='invalid_json'
    assert r['rows'][1]['failure_category']=='invalid_citation'
    assert r['rows'][1]['citations']==[] and r['overall']['withholding']==0
    assert r['overall']['completed_responses']==2


@pytest.mark.parametrize('category',['model_runtime_error','output_length_exhaustion','repetition'])
def test_partial_output_and_failure_denominators(document,passages,category):
    d=MockDecoder([OutputFailure(category,'do not print protected errors'),'{bad'])
    _,r=run(document,passages,d)
    row=r['rows'][0]
    assert row['raw_output']=='partial synthetic output' and row['failure_category']==category
    assert not row['response_completed'] and row['latency_seconds']>=0
    assert r['overall']['planned']==24 and r['overall']['withholding']==0
    assert len(d.calls)==(1 if category=='model_runtime_error' else 2)


def test_pair_preflight_blocks_both_before_decode(document,passages):
    d=MockDecoder();d.prompt_tokens=lambda messages:[1]*7809
    _,r=run(document,passages,d)
    assert not d.calls and r['overall']['attempted']==0
    assert all(x['status']=='blocked' for x in r['rows'][:2])


def test_provenance_and_question_mismatch_prevent_calls(document,passages):
    for altered in [passages[0].model_copy(update={'text':'tampered'}),passages[0].model_copy(update={'document_id':'wrong'})]:
        d=MockDecoder();_,r=run(document,[altered],d)
        assert not d.calls and r['rows'][0]['failure_category']=='source_validation_failure'
    d=MockDecoder();runner=PairedComparison(d,proposal())
    runner.pair('pilot-v01-in-06-en',Question(text='different question',language='en'),evidence(document,passages),[document])
    assert not d.calls


def test_readonly_e5_adapter_mocked(document,passages):
    p=proposal()
    five=[passages[0].model_copy(update={'passage_id':f'synthetic-{i}'}) for i in range(5)]
    service=SimpleNamespace(retriever=SimpleNamespace(cache_hit=True,cache_key=p.embedding_cache_key,
        model_fingerprint=p.retrieval_model_fingerprint,corpus_fingerprint=p.corpus_fingerprint,retriever_id=p.index_identifier),
        passages={x.passage_id:x for x in five},documents={document.document_id:document})
    calls=[]
    def search(payload):
        calls.append(payload)
        return {'mode':'local_e5','corpus_fingerprint':p.corpus_fingerprint,'results':[
            {'rank':i,'passage_id':x.passage_id,'document_id':x.document_id,'text':x.text,'evidence_span':x.evidence_span.model_dump()} for i,x in enumerate(five,1)]}
    service.search=search
    pack=E5EvidenceAdapter(service,p).retrieve(Question(text='Synthetic question',language='hi'))
    assert len(pack.passages)==5 and calls==[{'query':'Synthetic question','language':'hi'}]
    service.retriever.cache_hit=False
    with pytest.raises(ValueError):E5EvidenceAdapter(service,p)


def test_qwen_adapter_wraps_without_loading_or_decode(monkeypatch):
    from krishisaarthi.rag.two_stage import MLXTextBackend
    # Bypass constructor ONLY for mock tests; never load real weights.
    backend=MLXTextBackend.__new__(MLXTextBackend)
    backend.config=json.loads(Path('configs/mlx-generator.json').read_text())
    backend.tokenizer=SimpleNamespace(apply_chat_template=lambda m,**kw:[1,2,3])
    seen=[];backend.decode=lambda m,k:seen.append(k) or 'mock'
    adapter=QwenDecoder(backend,proposal())
    assert adapter.prompt_tokens([])==[1,2,3] and seen==[]
    assert adapter.decode([],384)=='mock' and seen==[384]
    with pytest.raises(ValueError):adapter.decode([],512)
    backend.config['inference']['temperature']=0.5
    with pytest.raises(ValueError):QwenDecoder(backend,proposal())


def test_telemetry_mismatch_stops_and_suppresses(document,passages):
    class Broken(MockDecoder):
        def decode(self,*a):
            raw=super().decode(*a);self.last_metrics['input_tokens']=999;return raw
    d=Broken();_,r=run(document,passages,d)
    assert len(d.calls)==1 and r['rows'][0]['failure_category']=='telemetry_failure'
    assert r['rows'][0]['claims']==[]


def test_private_writer(tmp_path):
    import subprocess
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'data').mkdir();(tmp_path/'.gitignore').write_text('data/\n')
    save_private(tmp_path,'synthetic.json',{'raw_output':'synthetic only'})
    f=tmp_path/'data/synthetic.json'
    assert f.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):save_private(tmp_path,'synthetic.json',{})
    with pytest.raises(ValueError):save_private(tmp_path,'../public.json',{})


def test_accounting_language_differences_and_citation_na(document,passages):
    _,r=run(document,passages)
    assert r['paired_prompt_tokens'][0]['rag_minus_direct']==40
    assert r['by_language']['hi']['direct']['planned']==4
    assert r['by_language']['en']['rag']['attempted']==1
    assert r['mock_decode_attempts']==2 and r['model_attempts']==0
    assert r['rows'][0]['citation_applicability']=='not_applicable'
    assert r['rows'][1]['citation_applicability']=='applicable'
    assert all(x['externally_verified_false'] is None for x in r['rows'])


def test_missing_completed_telemetry_is_failure(document,passages):
    class Missing(MockDecoder):
        def decode(self,*a):
            raw=super().decode(*a);self.last_metrics={};return raw
    d=Missing();_,r=run(document,passages,d)
    assert len(d.calls)==1 and r['rows'][0]['failure_category']=='telemetry_failure'
    assert r['rows'][0]['raw_output'] and r['rows'][0]['output_tokens'] is None


def test_import_does_not_load_model_libraries():
    import subprocess,sys,os
    code="import sys; from krishisaarthi.evaluation import qa_adapters; assert not any(n.split('.')[0] in {'mlx','mlx_lm','torch','transformers'} for n in sys.modules)"
    subprocess.run([sys.executable,'-c',code],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'src'},check=True)


def test_no_reference_or_label_input_fields():
    from pydantic import ValidationError
    for name in ('gold','reference_answer','answerable','expected_evidence'):
        with pytest.raises(ValidationError):Question.model_validate({'text':'synthetic','language':'en',name:'secret'})
