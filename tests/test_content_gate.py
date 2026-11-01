"""Synthetic controls; no corpus files, annotation labels or expected answers."""
import pytest
from krishisaarthi.schemas import Query
from krishisaarthi.retrieval.content_gate import ContentTokenPolicy, ContentGatedRRFRetriever
from krishisaarthi.retrieval.rrf import RRFRetriever
from test_rrf import Fake

Q=Query(query_id='synthetic-gate',text='?')


@pytest.mark.parametrize('kind',['positive','content'])
@pytest.mark.parametrize('scores',[[],[0.,0.],[-2.,0.]])
def test_empty_lexical_contribution_preserves_exact_dense_order(passages,kind,scores):
    lexical=Fake([0,1] if scores else [],scores)
    dense=Fake([1,0],[.4,.4])
    if kind=='content':r=ContentGatedRRFRetriever(lexical,dense,require_complete=False)
    else:r=RRFRetriever(lexical,dense,gate_nonpositive_lexical=True)
    r.index(passages)
    expected=dense.search(Q,2);actual=r.search(Q,2)
    assert [(h.passage_id,h.rank,h.document_id,h.evidence_span) for h in actual]==[
        (h.passage_id,h.rank,h.document_id,h.evidence_span) for h in expected]
    assert [h.score for h in actual]==[1/61,1/62]


def test_positive_punctuation_or_generic_overlap_cannot_contribute(passages):
    p=[x.model_copy(update={'text':'wheat the ?'}) for x in passages]
    r=ContentGatedRRFRetriever(Fake([0,1],[7,6]),Fake([1,0]),require_complete=False)
    r.index(p);q=Query(query_id='q',text='the wheat?')
    assert [h.passage_id for h in r.search(q,2)]==[p[i].passage_id for i in [1,0]]
    assert not any(x['lexical_contribution_allowed'] for x in r.gate_diagnostics(q))


@pytest.mark.parametrize('term',['PBW-999','12.5','sulphur','Happy Seeder','गेहूँ','NPK42'])
def test_entity_and_numeric_tokens_retained(term,passages):
    p=[x.model_copy(update={'text':term}) for x in passages]
    r=ContentGatedRRFRetriever(Fake([0,1],[1,0]),Fake([0,1]),require_complete=False)
    r.index(p);q=Query(query_id='q',text=term)
    rows=r.gate_diagnostics(q)
    assert rows[0]['shared_content_tokens'] and rows[0]['lexical_contribution_allowed']
    assert not rows[1]['lexical_contribution_allowed']
    assert r.search(q,2)[0].score==2/61


def test_hyphenation_numbers_and_generic_policy():
    p=ContentTokenPolicy()
    assert p.shared('PBW-999 rice-residue 12.5% wheat?', 'pbw 999 rice residue 12.5 wheat')==[
        '12','5','999','pbw','residue','rice']
    assert not p.shared('the wheat crop?', 'the wheat crop?')
    assert p.shared('20 kg','20 kg')==['20','kg']


def test_deterministic_ties_no_rank_compaction_and_no_mutation(passages):
    p=[x.model_copy(update={'text':'alpha'}) for x in passages]
    lex=Fake([0,1],[0,2]);den=Fake([0,1]);r=ContentGatedRRFRetriever(lex,den,require_complete=False)
    r.index(p);q=Query(query_id='q',text='alpha');before=lex.search(q,2)
    hits=r.search(q,2)
    assert hits==r.search(q,2) and before==lex.search(q,2)
    score={h.passage_id:h.score for h in hits}
    assert score[p[0].passage_id]==1/61 and score[p[1].passage_id]==2/62
    tied=ContentGatedRRFRetriever(Fake([0,1]),Fake([1,0]),require_complete=False);tied.index(p)
    assert [h.passage_id for h in tied.search(q,2)]==sorted(x.passage_id for x in p[:2])


def test_rejected_candidate_still_validates_provenance(passages):
    class Bad(Fake):
        def search(self,q,k):
            return [super().search(q,k)[0].model_copy(update={'document_id':'wrong'})]
    r=ContentGatedRRFRetriever(Bad([0],[0]),Fake([1]),require_complete=False);r.index(passages)
    with pytest.raises(ValueError,match='provenance'):r.search(Q,1)


def test_nonfinite_score_not_hidden_by_content_mask(passages):
    class Bad(Fake):
        def search(self,q,k):
            return [super().search(q,k)[0].model_copy(update={'score':float('nan')})]
    r=ContentGatedRRFRetriever(Bad([0]),Fake([1]),require_complete=False);r.index(passages)
    with pytest.raises(ValueError,match='lexical'):r.search(Q,1)
