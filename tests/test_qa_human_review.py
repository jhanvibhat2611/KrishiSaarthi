"""Synthetic-only QA-C4 validation. No models, sources or approved answers."""
import copy
import json
from pathlib import Path

import pytest
from krishisaarthi.evaluation.qa_human_review import (
    FIELDS, CITATION, blank_review, validate_records, validate_decisions,
    save_export, snapshot,
)


def fixture():
    cases=[]; rows=[]
    for i in range(12):
        cid=f'synthetic-{i}'; language=['en','hi','hinglish'][i%3]
        cases.append({'case_id':cid,'question':{'text':'SYNTHETIC question', 'language':language}})
        for arm in ('direct','rag'):
            status=('error' if i<4 else 'uncertain') if arm=='direct' else ('error' if i<2 else 'review_required' if i<8 else 'insufficient')
            rows.append({'case_id':cid,'arm':arm,'language':language,'attempted':True,'status':status,
                         'failure_category':'schema_failure' if status=='error' else None,
                         'raw_output':'SYNTHETIC malformed raw <script>never execute</script>',
                         'claims':[{'text':'SYNTHETIC claim','evidence_handles':['E1','E2']}] if status=='review_required' else [],
                         'citations':[], 'citation_applicability':'not_applicable' if arm=='direct' else 'applicable'})
    result={'rows':rows,'model_attempts':24,'reference_scoring_enabled':False}
    package={'package_fingerprint':'synthetic-fingerprint','binding':{'run_id':'synthetic'},'reviews':[blank_review(r) for r in rows]}
    payload={'package_fingerprint':'synthetic-fingerprint','reviewer_id':'synthetic-test-only','records':copy.deepcopy(package['reviews'])}
    return result, {'cases':cases}, package, payload


def test_exact_pairing_and_counts():
    r,h,_,_=fixture()
    assert validate_records(r,h)=={'direct':(12,8,0,8,4),'rag':(12,10,6,4,2)}


@pytest.mark.parametrize('mutation',['missing','duplicate','raw','language','promoted_failure','direct_citation','status','attempt','failure_category'])
def test_records_reject_inconsistency(mutation):
    r,h,_,_=fixture(); row=r['rows'][0]
    if mutation=='missing':r['rows'].pop()
    elif mutation=='duplicate':r['rows'][-1]=r['rows'][0]
    elif mutation=='raw':row['raw_output']=''
    elif mutation=='language':row['language']='wrong'
    elif mutation=='promoted_failure':row['claims']=[{}]
    elif mutation=='direct_citation':row['citations']=[{}]
    elif mutation=='status':row['status']='invented'
    elif mutation=='attempt':row['attempted']=False
    else:row['failure_category']=None
    with pytest.raises(ValueError):validate_records(r,h)


def test_initial_scores_blank_including_claims():
    _,_,p,_=fixture()
    for r in p['reviews']:
        assert all(r[k] is None for k in FIELDS)
        assert r['notes']=='' and r['reviewed'] is False
        assert all(c['support'] is None for c in r['claim_citations'])
        if r['arm']=='direct':assert r['citation_applicability']=='not_applicable' and r['claim_citations']==[]


def test_partial_export_no_mutation_private_and_unique(tmp_path):
    _,_,p,v=fixture(); before=copy.deepcopy(v)
    originals=tmp_path/'original';originals.mkdir();(originals/'results.json').write_text('SYNTHETIC unchanged')
    frozen=snapshot(originals)
    a=save_export(tmp_path,v,p);b=save_export(tmp_path,v,p)
    assert a!=b and v==before and snapshot(originals)==frozen
    path=tmp_path/'exports'/a; out=json.loads(path.read_text())
    assert out['records']==v['records'] and out['reviewed_count']==0
    assert out['reference_approval'] is False and out['externally_verified_false'] is None
    assert path.stat().st_mode&0o777==0o600
    assert (tmp_path/'exports').stat().st_mode&0o777==0o700


@pytest.mark.parametrize('mutation',['binding','reviewer','duplicate','missing','score','reviewed_blank','direct_citation','handle','claim_score','claim_missing','extra'])
def test_export_rejects_malformed_or_inferred_review(mutation):
    _,_,p,v=fixture(); r=v['records'][0]; claim=next(x for x in v['records'] if x['claim_citations'])
    if mutation=='binding':v['package_fingerprint']='different'
    elif mutation=='reviewer':v['reviewer_id']=' '
    elif mutation=='duplicate':v['records'][-1]=r
    elif mutation=='missing':v['records'].pop()
    elif mutation=='score':r['factual_support']='automatic perfect score'
    elif mutation=='reviewed_blank':r['reviewed']=True
    elif mutation=='direct_citation':r['citation_applicability']='valid'
    elif mutation=='handle':claim['claim_citations'][0]['handle']='E99'
    elif mutation=='claim_score':claim['claim_citations'][0]['support']='structurally valid therefore factual'
    elif mutation=='claim_missing':claim['claim_citations'].pop()
    else:v['gold']=True
    with pytest.raises(ValueError):validate_decisions(v,p)


def test_explicit_review_requires_every_claim_handle():
    _,_,p,v=fixture(); r=next(x for x in v['records'] if x['claim_citations'])
    for k,opts in FIELDS.items():r[k]=opts[-1]
    r['reviewed']=True;r['claim_citations'][0]['support']=CITATION[-1]
    with pytest.raises(ValueError):validate_decisions(v,p)
    r['claim_citations'][1]['support']=CITATION[-1]
    assert validate_decisions(v,p)==v


def test_template_accessibility_privacy_and_raw_failures():
    text=(Path(__file__).parents[1]/'src/krishisaarthi/evaluation/qa_review.html').read_text()
    for expected in ['grid-template-columns:1fr 1fr','@media(max-width:760px)', 'grid-template-columns:1fr', ':focus-visible', 'aria-live="polite"', 'Skip to response reviews', "node('pre',row.raw_output)", "row.status==='error'", 'Supplied-evidence citation validation: N/A', 'claim_citations', 'textContent', '— Select after inspection —']:
        assert expected in text
    assert 'innerHTML' not in text and 'localStorage' not in text and 'https://' not in text
    assert "fetch('/export'" in text
