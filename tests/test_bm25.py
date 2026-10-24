import pytest
from krishisaarthi.retrieval.bm25 import BM25Retriever
from krishisaarthi.schemas import Query


def test_ranking_and_stable_ties(passages):
    retriever = BM25Retriever()
    retriever.index(passages)
    assert 'seedmarker' in next(p.text for p in passages if p.passage_id == retriever.search(Query(query_id='q', text='seedmarker'), 1)[0].passage_id)
    query = Query(query_id='q', text='अज्ञात')
    first = retriever.search(query, 100)
    retriever.index(list(reversed(passages)))
    assert retriever.search(query, 100) == first
    assert [r.passage_id for r in first] == sorted(p.passage_id for p in passages)


def test_index_errors(passages):
    retriever = BM25Retriever()
    with pytest.raises(ValueError):
        retriever.search(Query(query_id='q', text='x'), 1)
    with pytest.raises(ValueError):
        retriever.index([])
    with pytest.raises(ValueError):
        retriever.index([passages[0], passages[0]])
    retriever.index(passages)
    with pytest.raises(ValueError):
        retriever.search(Query(query_id='q', text='x'), 0)
