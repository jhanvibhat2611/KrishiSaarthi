"""Synthetic handoff and subprocess tests only; never load research data or models."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from pydantic import ValidationError
from krishisaarthi.evaluation.qa_runtime import (
    PrivateHandoff,HandoffCase,proposal,validate_handoff,private_write,consume_synthetic,environment,
)
from krishisaarthi.evaluation.qa_comparison_fake import Question,fingerprint

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def document(tmp_path):
    from krishisaarthi.ingestion.plaintext import PlainTextIngestor
    from krishisaarthi.schemas import SourceMetadata
    f=tmp_path/'fixture.txt';f.write_text('SYNTHETIC: Invented tag Cedar.')
    return PlainTextIngestor().ingest(f,SourceMetadata(document_id='synthetic-c3-test'))


@pytest.fixture
def local_repo(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'.gitignore').write_text('data/\n');(tmp_path/'data').mkdir()
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    shutil.copy(ROOT/'configs/qa-readiness-e5.proposed.json',tmp_path/'configs')
    shutil.copy(ROOT/'scripts/qa_runtime.py',tmp_path/'scripts')
    (tmp_path/'src').symlink_to(ROOT/'src',target_is_directory=True)
    (tmp_path/'.venv-rag').symlink_to(Path(sys.executable).parent.parent,target_is_directory=True)
    return tmp_path


def cli(root,*args):
    return subprocess.run([sys.executable,str(root/'scripts/qa_runtime.py'),*args],cwd=root,
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=30)


def test_cross_environment_dry_run(local_repo):
    directory=local_repo/'data/dry'
    result=cli(local_repo,'dry-run','--directory',str(directory))
    assert result.returncode==0,result.stderr
    summary=json.loads(result.stdout)
    assert summary['planned']==summary['mock_decode_calls']==24 and summary['model_calls']==summary['retrieval_calls']==0
    assert summary['failed']==1 and summary['withholding']==1
    assert 'Cedar' not in result.stdout+result.stderr
    rows=json.loads((directory/'mock-results.json').read_text())['rows']
    assert len(rows)==24 and all(r['human_review']['approval'] is None for r in rows)
    assert all(r['externally_verified_false'] is None for r in rows)
    assert all(r['citation_applicability']=='not_applicable' for r in rows if r['arm']=='direct')
    assert 'sources' not in json.loads(rows[0]['prompt'][1]['content'])
    assert 'sources' in json.loads(rows[1]['prompt'][1]['content'])
    assert directory.stat().st_mode & 0o777==0o700
    assert all(p.stat().st_mode & 0o777==0o600 for p in directory.iterdir())
    assert cli(local_repo,'dry-run','--directory',str(directory)).returncode!=0


def test_real_mode_cannot_be_enabled(local_repo):
    r=cli(local_repo,'real')
    assert r.returncode!=0 and 'disabled' in r.stderr
    r=cli(local_repo,'real','--authorized')
    assert r.returncode!=0
    assert not list((local_repo/'data').iterdir())


def handoff(document,passages):
    p=proposal(ROOT)
    five=[passages[0].model_copy(update={'passage_id':f'synthetic-{i}'}) for i in range(5)]
    # Existing fixture begins with SYNTHETIC: and has genuine source offsets/files.
    h=PrivateHandoff(mode='synthetic',proposal_sha256=fingerprint(p.model_dump()),corpus_fingerprint='synthetic-only',
        documents=[document],cases=[HandoffCase(case_id=c.case_id,question=Question(text='Synthetic question',language=c.language),passages=five) for c in p.cases])
    return p,h


def test_roundtrip_provenance_and_scope(document,passages):
    p,h=handoff(document,passages)
    loaded=validate_handoff(json.loads(h.model_dump_json()),p)
    assert loaded==h
    assert loaded.cases[0].passages[0].evidence_span==passages[0].evidence_span


@pytest.mark.parametrize('change',['source','language','question_extra','duplicate','fingerprint','held'])
def test_handoff_rejects_mismatch(document,passages,change):
    p,h=handoff(document,passages);data=h.model_dump()
    if change=='source':data['cases'][0]['passages'][0]['text']='SYNTHETIC: tampered'
    if change=='language':data['cases'][0]['question']['language']='hi'
    if change=='question_extra':data['cases'][0]['question']['reference_answer']='secret'
    if change=='duplicate':data['cases'][0]['passages'][1]=data['cases'][0]['passages'][0]
    if change=='fingerprint':data['proposal_sha256']='bad'
    if change=='held':data['cases'][0]['passages'][0]['document_id']='held-assam'
    with pytest.raises(ValueError):validate_handoff(data,p)


def test_checksum_and_permission_rejection(local_repo,document,passages):
    _,h=handoff(document,passages);directory=local_repo/'data/check';directory.mkdir(mode=0o700)
    f=directory/'handoff.json';private_write(f,h.model_dump())
    with pytest.raises(ValueError,match='checksum'):consume_synthetic(local_repo,directory,'wrong')
    f.chmod(0o644)
    with pytest.raises(ValueError,match='permissions'):consume_synthetic(local_repo,directory,hashlib.sha256(f.read_bytes()).hexdigest())


def test_environment_metadata_only():
    import importlib.metadata
    result=environment(Path(sys.executable),{'pydantic':importlib.metadata.version('pydantic')})
    assert result['packages']['pydantic']
    with pytest.raises(ValueError):environment(Path(sys.executable),{'pydantic':'0.0.invalid'})


def test_ui_is_not_imported_or_changed_by_dry_run(local_repo):
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/krishisaarthi/ui').rglob('*') if p.is_file()}
    assert cli(local_repo,'dry-run','--directory',str(local_repo/'data/ui-check')).returncode==0
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==v for p,v in before.items())


def test_oversize_handoff_keeps_planned_denominator_without_calls(local_repo,document,passages):
    _,h=handoff(document,passages);data=h.model_dump()
    for case in data['cases']:case['question']['text']='SYNTHETIC ' * 1000
    directory=local_repo/'data/oversize';directory.mkdir(mode=0o700)
    f=directory/'handoff.json';private_write(f,data)
    result=consume_synthetic(local_repo,directory,hashlib.sha256(f.read_bytes()).hexdigest())
    assert result['mock_decode_calls']==0 and result['planned']==result['unattempted']==result['blocked_slots']==24
    assert result['withholding']==0


def test_reference_fields_cannot_enter_handoff(document,passages):
    p,h=handoff(document,passages);data=h.model_dump();data['reference_answers']=['secret']
    with pytest.raises(ValidationError):validate_handoff(data,p)
