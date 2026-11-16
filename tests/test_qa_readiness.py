"""Fake-only readiness checks; no research inputs or model imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from pydantic import ValidationError
from krishisaarthi.evaluation.qa_readiness import (
    E5ComparisonProposal, readiness_manifest, SimulatedTelemetry, record_synthetic_pair,
)
from krishisaarthi.evaluation.qa_comparison_fake import Question, SyntheticCase, SyntheticEvidence, FakeFailure


def proposal():
    return E5ComparisonProposal.model_validate_json(Path('configs/qa-readiness-e5.proposed.json').read_text(),strict=True)


def fixture():
    return SyntheticCase(case_id='synthetic-ready',question=Question(text='Which invented tag?',language='hi'),
        retrieved_evidence=[SyntheticEvidence(passage_id='synthetic-e',document_id='demo-d',text='SYNTHETIC: tag Aspen.')])


def withholding():
    return json.dumps({'status':'uncertain','claims':[],'reason':'Canned uncertainty'})


def test_e5_proposal_and_pending_matrix():
    p=proposal();m=readiness_manifest(p)
    assert p.selected_retriever=='e5' and p.proposed_top_k==5
    assert len(p.cases)==12 and p.generation_calls_per_question_arm==1
    assert p.model_revision==p.tokenizer_revision
    assert m['real_model_calls']==0 and m['reference_scoring_enabled'] is False
    assert m['configuration']['real_execution_authorized'] is False
    assert m['input_budget_policy']['maximum_prompt_tokens_per_arm']==7808
    assert m['maximum_proposed_generated_tokens']==9216
    assert m['planned_accounting_no_execution']['overall']['primary_completion_denominator']==24
    assert m['planned_accounting_no_execution']['overall']['not_attempted_slots']==24
    assert all(m[k] is None for k in ['reference_approval','architecture_approval','execution_approval'])


@pytest.mark.parametrize('update',[
    {'selected_retriever':'bm25'}, {'proposed_top_k':3}, {'generation_calls_per_question_arm':2},
    {'index_identifier':'bm25'}, {'real_execution_authorized':True}, {'generation_packages':{}},
    {'gold':'secret'}, {'reference_answer':'secret'}, {'max_output_tokens_per_call':512},
])
def test_proposal_rejects_drift(update):
    with pytest.raises(ValidationError): E5ComparisonProposal.model_validate({**proposal().model_dump(),**update},strict=True)


def test_unequal_input_budgets_are_explicit_and_fake():
    r=record_synthetic_pair(fixture(),withholding(),withholding(),
       SimulatedTelemetry(input_tokens=42,output_tokens=10,runtime_seconds=0.1,finish_reason='stop'),
       SimulatedTelemetry(input_tokens=142,output_tokens=10,runtime_seconds=0.2,finish_reason='stop'))
    assert r['input_budget_comparison']['rag_minus_direct_tokens']==100
    assert r['planned_generation_budget']['max_output_tokens_per_arm']==384
    assert r['real_model_calls']==0
    for row in r['results']:
        assert row['input_tokens'] is row['output_tokens'] is row['generation_runtime_seconds'] is None
        assert row['raw_output']==withholding()
        assert all(v is None for k,v in row['human_review'].items() if k!='review_status')
    assert 'Aspen' not in json.dumps(r['results'][0]['prompt'])
    assert 'Aspen' in json.dumps(r['results'][1]['prompt'])
    assert r['results'][0]['question_sha256']==r['results'][1]['question_sha256']


@pytest.mark.parametrize('category,finish',[('model_runtime_error','runtime_error'),('output_length_exhaustion','length'),('repetition','repetition')])
def test_partial_failures_retained_not_abstentions(category,finish):
    r=record_synthetic_pair(fixture(),FakeFailure(category=category,raw_output='partial raw'),withholding(),
        SimulatedTelemetry(finish_reason=finish),SimulatedTelemetry())
    row=r['results'][0]
    assert row['raw_output']=='partial raw' and row['failure_category']==category
    assert r['input_budget_comparison']['rag_minus_direct_tokens'] is None
    totals=r['synthetic_attempt_accounting']['overall']
    assert totals['primary_completion_denominator']==2 and totals['failure_suppressed_answers']==1
    assert totals['deliberate_withholding']==1 and totals['simulated_completed_responses']==1


@pytest.mark.parametrize('update',[{'input_tokens':-1},{'output_tokens':385},{'runtime_seconds':float('nan')},{'runtime_seconds':-1.0},{'provenance':'MEASURED'}])
def test_invalid_telemetry(update):
    with pytest.raises(ValidationError): SimulatedTelemetry(**update)


def test_budget_boundary_and_inconsistent_finish():
    record_synthetic_pair(fixture(),withholding(),withholding(),SimulatedTelemetry(input_tokens=7808),SimulatedTelemetry())
    with pytest.raises(ValueError,match='budget'):
        record_synthetic_pair(fixture(),withholding(),withholding(),SimulatedTelemetry(input_tokens=7809),SimulatedTelemetry())
    with pytest.raises(ValueError,match='Finish'):
        record_synthetic_pair(fixture(),withholding(),withholding(),SimulatedTelemetry(finish_reason='length'),SimulatedTelemetry())


def test_preparation_has_no_model_imports_or_reference_access():
    code="""
import runpy,sys,json,builtins,pathlib
original=pathlib.Path.open
blocked=('data/',)
def guarded(self,*a,**kw):
    if '/data/' in str(self) or str(self).startswith('data/'):
        raise AssertionError('Preparation must not read research data')
    return original(self,*a,**kw)
pathlib.Path.open=guarded
m=runpy.run_path('scripts/prepare_qa_readiness.py');result=m['prepare']()
assert not any(n.split('.')[0] in {'mlx','mlx_lm','torch','transformers'} for n in sys.modules)
assert len(result['synthetic_examples'])==3
assert len(result['human_review']['rows'])==24
assert all(r['human_review']['approval'] is None for r in result['human_review']['rows'])
print('fake-only preparation passed')
"""
    r=subprocess.run([sys.executable,'-c',code],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'src'},capture_output=True,text=True)
    assert r.returncode==0,r.stderr
