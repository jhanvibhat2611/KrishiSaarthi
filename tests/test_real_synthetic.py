"""Adapter checks use fake decoding only; no actual model invocation."""
import json
import pytest
from krishisaarthi.ui.real_synthetic import SyntheticSource, messages, generate_once, CONTRACT_V1, CONTRACT_V2, SYSTEM_V1
from krishisaarthi.rag.handles import OutputFailure
SOURCE=SyntheticSource(passage_id='synthetic-check',document_id='demo-check',text='SYNTHETIC: A label is Cedar.',section='Test')
class Backend:
 def __init__(self,value):self.value=value;self.calls=0;self.last_metrics={};self.last_output_text=''
 def decode(self,prompt,max_tokens):
  self.calls+=1
  if isinstance(self.value,Exception):raise self.value
  self.last_output_text=self.value;return self.value

def run(value,language='en',contract_version=CONTRACT_V2):
 b=Backend(value);r=generate_once(b,{'query_id':'test','question':'Any newly worded question','language':language},[SOURCE],contract_version=contract_version);assert b.calls==1;return r

@pytest.mark.parametrize('language',['en','hi','hinglish'])
def test_supported_without_question_or_answer_allowlist(language):
 r=run(json.dumps({'status':'draft','claims':[{'text':'A freely generated claim.','evidence_handles':['E1']}],'reason':''}),language)
 assert r['response']['status']=='review_required'
 assert r['response']['claims'][0]['citations'][0]['document_id']=='demo-check'
 assert r['response']['factual_verification']=='pending_review'

@pytest.mark.parametrize('raw,category',[
 ('bad json','invalid_json'),
 ('{}','schema_failure'),
 (json.dumps({'status':'draft','claims':[],'reason':''}),'incorrect_status'),
 (json.dumps({'status':'draft','claims':[{'text':'x','evidence_handles':['E99']}],'reason':''}),'invalid_citation'),
 (json.dumps({'status':'draft','claims':[{'text':'x','evidence_handles':[]}],'reason':''}),'schema_failure'),
 (OutputFailure('output_length_exhaustion','cap'),'output_length_exhaustion'),
 (RuntimeError('private runtime details'),'model_runtime_error'),
])
def test_failures(raw,category):
 r=run(raw)['response'];assert r['status']=='error';assert r['failure_category']==category;assert r['answerability']=='not_established';assert r['answer'] is None

@pytest.mark.parametrize('status',['insufficient','uncertain'])
def test_abstention(status):
 r=run(json.dumps({'status':status,'claims':[],'reason':'Missing fact.'}))['response']
 assert r['status']==('UNANSWERABLE' if status=='insufficient' else 'uncertain')
 assert not r['claims'] and r['answer'] is None

def test_real_source_rejected():
 with pytest.raises(ValueError):messages('x','en',[SOURCE.model_copy(update={'text':'Not synthetic'})])

def test_nonempty_draft_reason_rejected_as_frozen_contract():
    r=run(json.dumps({'status':'draft','claims':[{'text':'A label is Cedar.','evidence_handles':['E1']}],'reason':'Extra explanation.'}),contract_version=CONTRACT_V1)
    assert r['response']['failure_category']=='incorrect_status'
    assert r['response']['status']=='error'

def test_failure_raw_output_preserved():
    raw='{"status":"draft"} trailing text'
    r=run(raw)
    assert r['raw_output']==raw and r['response']['failure_category']=='invalid_json'
    assert r['repair_attempts']==0 and r['retries']==0


def test_v2_explanation_is_preserved_as_metadata_not_answer():
    raw=json.dumps({'status':'draft','claims':[{'text':'A label is Cedar.','evidence_handles':['E1']}],'reason':'This label is recorded in the fictional note.'})
    r=run(raw)
    assert r['response']['contract_version']==CONTRACT_V2
    assert r['response']['status']=='review_required'
    assert r['response']['answer']=='A label is Cedar.'
    assert r['response']['reason']=='This label is recorded in the fictional note.'
    assert r['response']['reason_role']=='unverified_metadata_not_answer'
    assert r['response']['factual_verification']=='pending_review'
    assert r['response']['confidence']=='not_estimated'
    assert r['raw_output']==raw

@pytest.mark.parametrize('payload,category',[
    ({'status':'answered','claims':[],'reason':'x'},'schema_failure'),
    ({'status':'insufficient','claims':[{'text':'Invented answer','evidence_handles':['E1']}],'reason':'Missing fact'},'incorrect_status'),
    ({'status':'uncertain','claims':[{'text':'Invented answer','evidence_handles':['E1']}],'reason':'Unsure'},'incorrect_status'),
    ({'status':'insufficient','claims':[],'reason':'Missing','answer':'Invented answer'},'schema_failure'),
    ({'status':'insufficient','claims':[],'reason':'   '},'incorrect_status'),
    ({'status':'draft','claims':[{'text':'x','evidence_handles':['E1','E1']}],'reason':'Explanation'},'invalid_citation'),
    ({'status':'draft','claims':[{'text':'x','evidence_handles':['e1']}],'reason':'Explanation'},'invalid_citation'),
    ({'status':'draft','claims':[{'text':'x'}],'reason':'Explanation'},'schema_failure'),
    ({'status':'draft','claims':[{'text':'x','evidence_handles':['E1']}],'reason':None},'schema_failure'),
])
def test_v2_keeps_strict_failures(payload,category):
    r=run(json.dumps(payload))['response']
    assert r['status']=='error' and r['failure_category']==category
    assert r['answer'] is None and not r['claims']
    assert r['answerability']=='not_established'


def test_structural_acceptance_is_not_entailment():
    r=run(json.dumps({'status':'draft','claims':[{'text':'An unsupported invented fact.','evidence_handles':['E1']}],'reason':'An unsupported explanation.'}))['response']
    # This adapter cannot semantically verify arbitrary generated text. No certification.
    assert r['status']=='review_required' and r['factual_verification']=='pending_review'
    assert r['claims'][0]['support_verification']=='pending_review'
    assert r['reason_role']=='unverified_metadata_not_answer'


def test_versioned_prompt_and_unknown_version():
    v1,_=messages('Question','en',[SOURCE],CONTRACT_V1)
    v2,_=messages('Question','en',[SOURCE],CONTRACT_V2)
    assert v1[0]['content']==SYSTEM_V1
    assert v2[0]['content']!=v1[0]['content']
    assert 'reason may be empty or a short explanatory note' in v2[0]['content']
    with pytest.raises(ValueError):messages('Question','en',[SOURCE],'unknown')


@pytest.mark.parametrize('raw',[
    '{"status":"draft","status":"draft","claims":[{"text":"x","evidence_handles":["E1"]}],"reason":""}',
    '{"status":"insufficient","status":"draft","claims":[{"text":"x","evidence_handles":["E1"]}],"reason":""}',
    '{"status":"insufficient","claims":[],"claims":[],"reason":"Missing"}',
    '{"status":"insufficient","claims":[],"reason":"Missing","reason":"Missing"}',
    '{"status":"draft","claims":[{"text":"x","text":"x","evidence_handles":["E1"]}],"reason":""}',
    '{"status":"draft","claims":[{"text":"x","evidence_handles":["E99"],"evidence_handles":["E1"]}],"reason":""}',
    # Decoded key equality also rejects escaped duplicates.
    r'{"status":"draft","st\u0061tus":"draft","claims":[],"reason":""}',
    # Even duplicates inside an unknown nested object fail before schema parsing.
    '{"status":"insufficient","claims":[],"reason":"Missing","extra":{"items":[{"x":1,"x":1}]}}',
])
def test_v2_rejects_all_duplicate_object_keys(raw):
    r=run(raw)
    assert r['response']['failure_category']=='invalid_json'
    assert r['response']['status']=='error'
    assert r['response']['answer'] is None and not r['response']['claims']
    assert r['raw_output']==raw
    assert r['repair_attempts']==r['retries']==0


@pytest.mark.parametrize('count',[1,2,3])
def test_versioned_claim_count_boundary(count):
    raw=json.dumps({'status':'draft','claims':[
        {'text':f'Claim {i}','evidence_handles':['E1']} for i in range(count)
    ],'reason':''})
    v1=run(raw,contract_version=CONTRACT_V1)['response']
    v2=run(raw,contract_version=CONTRACT_V2)['response']
    assert v1['status']=='review_required'  # Historical schema permits three.
    if count<=2:
        assert v2['status']=='review_required' and len(v2['claims'])==count
        assert v2['factual_verification']=='pending_review'
    else:
        assert v2['status']=='error' and v2['failure_category']=='schema_failure'
        assert v2['answer'] is None


def test_v1_duplicate_key_behavior_is_not_retroactively_changed():
    raw='{"status":"insufficient","status":"draft","claims":[{"text":"x","evidence_handles":["E1"]}],"reason":""}'
    v1=run(raw,contract_version=CONTRACT_V1)
    assert v1['response']['status']=='review_required'
    assert v1['raw_output']==raw
    assert run(raw)['response']['failure_category']=='invalid_json'
