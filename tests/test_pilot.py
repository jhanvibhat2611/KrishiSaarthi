"""Synthetic fixtures only; selection authorization here is simulated test data."""
import copy
import pytest
from test_corpus import corpus_doc, region
from krishisaarthi.config import ChunkingConfig
from krishisaarthi.corpus.review import review_fingerprint
from krishisaarthi.corpus.pilot import build_selected_pilot


def selection(r):
    return dict(version='synthetic', authorization_basis='explicit_user_selection',
        independent_agricultural_correctness='not_established', current_recommendation_validity='not_established',
        expected_region_count=1, excluded_region_ids=['held-synthetic'], applicability='synthetic only',
        selected_regions=[dict(region_id=r.region_id, review_fingerprint=review_fingerprint(r),
            evidence_span=r.evidence_span.model_dump(), source_matching_reference='synthetic-test-only',
            conditions_and_limitations='synthetic conditions')])


def test_user_selection_does_not_fabricate_review(corpus_doc):
    r=region(corpus_doc); before=r.model_dump_json(); s=selection(r)
    first,h=build_selected_pilot([corpus_doc],[r],[],ChunkingConfig(size=5,overlap=1),s)
    second,hh=build_selected_pilot([corpus_doc],[r],[],ChunkingConfig(size=5,overlap=1),s)
    assert first==second and h==hh and r.model_dump_json()==before
    assert r.status=='pending' and r.reviewer is None and not r.checks.consistency_clear
    assert all('reviewer' not in p.domain_metadata for p in first)
    assert all(p.domain_metadata['independent_agricultural_correctness']=='not_established' for p in first)
    assert all(p.text==corpus_doc.canonical_text[p.evidence_span.start_char:p.evidence_span.end_char] for p in first)


@pytest.mark.parametrize('change',['count','duplicate','excluded','substitution','fingerprint','span','authorization','correctness','context','extra_region'])
def test_selection_rejects_invalid_admission(corpus_doc,change):
    r=region(corpus_doc); s=selection(r); rows=[r]
    if change=='count':s['expected_region_count']=2
    elif change=='duplicate':s['selected_regions']*=2;s['expected_region_count']=2
    elif change=='excluded':s['excluded_region_ids']=[r.region_id]
    elif change=='substitution':r.region_id='original-parent-not-selected'
    elif change=='fingerprint':s['selected_regions'][0]['review_fingerprint']='wrong'
    elif change=='span':s['selected_regions'][0]['evidence_span']['end_char']-=1
    elif change=='authorization':s['authorization_basis']='automatic'
    elif change=='correctness':s['independent_agricultural_correctness']='verified'
    elif change=='context':s['selected_regions'][0]['conditions_and_limitations']=''
    elif change=='extra_region':
        extra=copy.deepcopy(r);extra.region_id='held-synthetic';rows.append(extra)
    with pytest.raises(ValueError):build_selected_pilot([corpus_doc],rows,[],ChunkingConfig(),s)


def test_selection_cannot_bypass_audit_issue(corpus_doc):
    from krishisaarthi.corpus.models import Issue
    r=region(corpus_doc)
    i=Issue(issue_id='synthetic',code='privacy',document_id=corpus_doc.document_id,physical_page=1,start_char=0,end_char=10,detail='Synthetic issue')
    with pytest.raises(ValueError,match='Unresolved audit'):
        build_selected_pilot([corpus_doc],[r],[i],ChunkingConfig(),selection(r))


def test_frozen_loader_roundtrip_and_tamper_detection(corpus_doc, tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from krishisaarthi.config import load_config
    from krishisaarthi.retrieval.bm25 import BM25Retriever
    from krishisaarthi.corpus.pilot import bm25_state, load_pilot
    r = region(corpus_doc)
    selected = selection(r)
    config_text = (Path(__file__).parents[1] / 'configs/baseline.toml').read_text()
    (tmp_path / 'config.toml').write_text(config_text)
    config = load_config(tmp_path / 'config.toml')
    passages, _ = build_selected_pilot([corpus_doc], [r], [], config.chunking, selected)
    (tmp_path / 'passages.jsonl').write_text(''.join(p.model_dump_json()+'\n' for p in passages))
    (tmp_path / 'selection.json').write_text(json.dumps(selected))
    (tmp_path / 'manifest.json').write_text('{}')
    retriever = BM25Retriever(config.bm25)
    retriever.index(passages)
    (tmp_path / 'bm25-index.json').write_text(json.dumps(bm25_state(retriever)))
    ledger = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    (tmp_path / 'integrity.json').write_text(json.dumps(ledger))
    loaded, rebuilt = load_pilot(tmp_path)
    assert loaded == passages and bm25_state(rebuilt) == bm25_state(retriever)
    (tmp_path / 'passages.jsonl').write_text('')
    with pytest.raises(ValueError, match='artifact changed'):
        load_pilot(tmp_path)
