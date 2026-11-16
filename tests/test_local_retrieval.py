"""Synthetic cache-only boundary tests. Never read the research corpus."""
import json
import pytest
from test_dense import FakeEncoder, SPECS
from krishisaarthi.ui.local import CachedE5, ledger
from krishisaarthi.retrieval.e5 import E5Retriever

@pytest.fixture
def fifteen(passages):
    return [passages[0].model_copy(update={'passage_id':f'synthetic-{i:02}'}) for i in range(15)]

def test_missing_cache_never_encodes_or_writes(fifteen,tmp_path):
    enc=FakeEncoder(768)
    r=CachedE5(enc,SPECS['e5'],'synthetic',tmp_path)
    with pytest.raises(FileNotFoundError): r.index(fifteen)
    assert not enc.seen and not list(tmp_path.iterdir())

def test_cache_only_validation(fifteen,tmp_path):
    e=FakeEncoder(768)
    writer=E5Retriever(e,SPECS['e5'],'synthetic',tmp_path);writer.index(fifteen)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    enc=FakeEncoder(768);r=CachedE5(enc,SPECS['e5'],'synthetic',tmp_path);r.index(fifteen)
    assert r.cache_hit and not enc.seen
    assert before=={p:p.read_bytes() for p in before}
    meta=tmp_path/r.cache_key/'manifest.json';data=json.loads(meta.read_text());data['passage_ids'].reverse();meta.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='provenance'):r.index(fifteen)
    data['passage_ids'].reverse();meta.write_text(json.dumps(data))
    (tmp_path/r.cache_key/'embeddings.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes'):r.index(fifteen)

def test_wrong_collection_rejected(fifteen,tmp_path):
    r=CachedE5(FakeEncoder(768),SPECS['e5'],'synthetic',tmp_path)
    with pytest.raises(ValueError):r.index(fifteen[:14])
    with pytest.raises(ValueError):r.index([fifteen[0]]*15)

def test_ledger_rejects_missing_and_changed(tmp_path):
    (tmp_path/'integrity.json').write_text(json.dumps({'missing':'abc'}))
    with pytest.raises(FileNotFoundError):ledger(tmp_path)
    (tmp_path/'missing').write_text('bad')
    with pytest.raises(ValueError):ledger(tmp_path)
