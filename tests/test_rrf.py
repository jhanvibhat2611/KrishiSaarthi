"""Synthetic RRF tests; no research data or model files."""
import pytest
from krishisaarthi.schemas import Query, RetrievalResult
from krishisaarthi.retrieval.rrf import RRFRetriever

class Fake:
    retriever_id='synthetic'
    def __init__(self, order, scores=None):
        self.order=order; self.scores=scores; self.calls=[]
    def index(self, passages): self.passages=passages
    def search(self,q,k):
        self.calls.append(k)
        return [RetrievalResult(query_id=q.query_id,passage_id=self.passages[i].passage_id,
            document_id=self.passages[i].document_id,evidence_span=self.passages[i].evidence_span,
            rank=r,score=self.scores[r-1] if self.scores else 1.,retriever_id=self.retriever_id)
            for r,i in enumerate(self.order[:k],1)]

Q=Query(query_id='synthetic',text='wheat')

def test_scores_full_depth_provenance(passages):
    a,b=Fake([0,1]),Fake([1,0]);h=RRFRetriever(a,b);h.index(passages)
    hits=h.search(Q,1)
    assert a.calls==b.calls==[len(passages)]
    assert hits[0].score==pytest.approx(1/61+1/62)
    assert hits[0].passage_id==min(p.passage_id for p in passages[:2])
    p=next(p for p in passages if p.passage_id==hits[0].passage_id)
    assert hits[0].evidence_span==p.evidence_span and hits[0].document_id==p.document_id
    assert hits[0].retriever_id==h.retriever_id

@pytest.mark.parametrize('score',[0.,-1.])
def test_gating_and_no_rank_compaction(passages,score):
    h=RRFRetriever(Fake([0,1],[score,2.]),Fake([0,1]),gate_nonpositive_lexical=True)
    h.index(passages);scores={r.passage_id:r.score for r in h.search(Q,2)}
    assert scores[passages[0].passage_id]==pytest.approx(1/61)
    assert scores[passages[1].passage_id]==pytest.approx(2/62)
    standard=RRFRetriever(Fake([0],[score]),Fake([]));standard.index(passages)
    assert standard.search(Q,1)[0].score==pytest.approx(1/61)

def test_all_gated_equals_dense_order(passages):
    h=RRFRetriever(Fake([0,1],[0.,0.]),Fake([1,0]),gate_nonpositive_lexical=True)
    h.index(passages)
    assert [r.passage_id for r in h.search(Q,2)]==[passages[i].passage_id for i in [1,0]]

def test_duplicates_rejected_and_missing_contributes_zero(passages):
    h=RRFRetriever(Fake([0,0]),Fake([1]));h.index(passages)
    with pytest.raises(ValueError,match='Duplicate'):h.search(Q,2)
    h=RRFRetriever(Fake([]),Fake([1]));h.index(passages)
    assert h.search(Q,5)[0].score==pytest.approx(1/61)
    assert len(h.search(Q,5))==1
    h=RRFRetriever(Fake([]),Fake([1]),require_complete=True);h.index(passages)
    with pytest.raises(ValueError,match='Complete'):h.search(Q,1)

def test_invalid_provenance(passages):
    class Bad(Fake):
        def search(self,q,k):
            return [super().search(q,k)[0].model_copy(update={'document_id':'wrong'})]
    h=RRFRetriever(Bad([0]),Fake([1]));h.index(passages)
    with pytest.raises(ValueError,match='provenance'):h.search(Q,1)

@pytest.mark.parametrize('constant',[0,-1,float('nan'),float('inf'),True])
def test_invalid_constant(constant):
    with pytest.raises(ValueError):RRFRetriever(Fake([]),Fake([]),constant)

def test_configurable_constant_and_state(passages):
    h=RRFRetriever(Fake([0]),Fake([0]),constant=10)
    with pytest.raises(ValueError):h.search(Q,1)
    with pytest.raises(ValueError):h.index([])
    with pytest.raises(ValueError):h.index([passages[0],passages[0]])
    h.index(passages)
    assert h.search(Q,1)[0].score==pytest.approx(2/11)
    with pytest.raises(ValueError):h.search(Q,0)
