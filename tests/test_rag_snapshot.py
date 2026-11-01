"""Query/corpus binding for sequential encoder/generator execution."""
import runpy
from pathlib import Path
import pytest
from krishisaarthi.config import fingerprint
from krishisaarthi.schemas import Query
from krishisaarthi.retrieval.bm25 import BM25Retriever

Snapshot=runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_rag_reliability.py'))['RankingSnapshot']

@pytest.fixture
def snapshot(passages):
    q=Query(query_id='synthetic',text='synthetic')
    r=BM25Retriever();r.index(passages)
    data={'passages_fingerprint':fingerprint([p.model_dump() for p in passages]),'retrievers':{'bm25':{
        'retriever_id':r.retriever_id,'queries':{q.query_id:{'query':q.model_dump(),'ranking':[h.model_dump() for h in r.search(q,99)]}}}}}
    return Snapshot(data,'bm25'),q

def test_snapshot_preserves_ranking(snapshot,passages):
    s,q=snapshot;s.index(passages)
    assert [h.rank for h in s.search(q,2)]==[1,2]
    assert all(h.passage_id in {p.passage_id for p in passages} for h in s.search(q,2))

def test_snapshot_rejects_corpus_change(snapshot,passages):
    with pytest.raises(ValueError,match='corpus mismatch'):snapshot[0].index(passages[:-1])

def test_snapshot_rejects_changed_question(snapshot):
    s,q=snapshot
    with pytest.raises(ValueError,match='query mismatch'):s.search(q.model_copy(update={'text':'different synthetic request'}),5)
