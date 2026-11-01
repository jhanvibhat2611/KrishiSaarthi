"""Synthetic lexical controls; no agricultural source passages or signed queries."""
import pytest
from rank_bm25 import BM25Okapi
from krishisaarthi.schemas import Query, SourceMetadata
from krishisaarthi.config import ChunkingConfig
from krishisaarthi.ingestion.plaintext import PlainTextIngestor
from krishisaarthi.preprocessing.chunking import DeterministicChunker
from krishisaarthi.retrieval.bm25 import BM25Retriever
from krishisaarthi.retrieval.punctuation_filtered import (
    PunctuationFilteredTokenizer, PunctuationFilteredBM25Retriever)
from krishisaarthi.retrieval.rrf import RRFRetriever


@pytest.fixture
def synthetic(tmp_path):
    result=[]
    for i,text in enumerate(['alpha crop? PBW-999 12.5%!', 'beta soil?', 'gamma rain?']):
        p=tmp_path/f'{i}.txt';p.write_text(text)
        d=PlainTextIngestor().ingest(p,SourceMetadata(document_id=f'synthetic-{i}'))
        result.extend(DeterministicChunker(ChunkingConfig()).chunk(d))
    return result


def test_terms_preserve_entities_numbers_and_attached_marks():
    assert PunctuationFilteredTokenizer().terms('PBW-999 rice-residue 12.5% गेहूँ NPK42 । ?') == [
        'pbw','999','rice','residue','12','5','गेहूँ','npk42']


@pytest.mark.parametrize('text',['?!—।','!!!','\u0301','💧'])
def test_punctuation_and_symbols_cannot_score(text,synthetic):
    r=PunctuationFilteredBM25Retriever();r.index(synthetic);q=Query(query_id='q',text=text)
    assert all(x.score==0 for x in r.search(q,3))
    assert r.diagnostics(q,3)['no_positive_match']
    assert r.diagnostics(q,3)['query_token_count']==0


def test_unknown_word_with_shared_punctuation_cannot_score(synthetic):
    q=Query(query_id='q',text='अनुपस्थित?')
    old=BM25Retriever();old.index(synthetic)
    new=PunctuationFilteredBM25Retriever();new.index(synthetic)
    assert any(r.score>0 for r in old.search(q,3))
    assert all(r.score==0 for r in new.search(q,3))


def test_index_and_query_filter_consistently(synthetic):
    r=PunctuationFilteredBM25Retriever();r.index(synthetic)
    q=Query(query_id='q',text='alpha?!')
    plain=Query(query_id='q',text='alpha')
    assert r.search(q,3)==r.search(plain,3)
    assert r.search(q,3)[0].score>0
    ordered=sorted(synthetic,key=lambda p:p.passage_id)
    t=PunctuationFilteredTokenizer()
    expected=BM25Okapi([t.terms(p.text) for p in ordered]).get_scores(t.terms(q.text))
    scores={p.passage_id:float(s) for p,s in zip(ordered,expected)}
    assert all(h.score==scores[h.passage_id] for h in r.search(q,3))


def test_original_identity_and_behavior_unchanged(synthetic):
    old=BM25Retriever();before=old.retriever_id;old.index(synthetic)
    q=Query(query_id='q',text='?');scores=old.search(q,3)
    new=PunctuationFilteredBM25Retriever();new.index(synthetic)
    assert old.retriever_id==before and new.retriever_id!=before
    assert old.search(q,3)==scores and '?' in old.tokenizer.terms('?')


def test_ties_and_provenance_with_rrf(synthetic):
    lexical=PunctuationFilteredBM25Retriever();dense_stub=BM25Retriever()
    r=RRFRetriever(lexical,dense_stub,constant=60,require_complete=True);r.index(synthetic)
    q=Query(query_id='q',text='unseen')
    hits=r.search(q,3)
    assert [h.passage_id for h in hits]==sorted(p.passage_id for p in synthetic)
    for h in hits:
        p=next(p for p in synthetic if p.passage_id==h.passage_id)
        assert h.evidence_span==p.evidence_span and h.document_id==p.document_id
        assert h.score==2/(60+h.rank)


def test_empty_filtered_passage_rejected(synthetic):
    p=synthetic[0].model_copy(update={'text':'???'})
    with pytest.raises(ValueError,match='tokens'):
        PunctuationFilteredBM25Retriever().index([p])
