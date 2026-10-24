import pytest
from krishisaarthi.schemas import EvidenceSpan, GoldLabel, Query, RetrievalResult
from krishisaarthi.evaluation.metrics import fully_covered, score_at_k
from krishisaarthi.evaluation.runner import evaluate
from krishisaarthi.evaluation.labels import validate_labels
from krishisaarthi.config import EvaluationConfig
from krishisaarthi.retrieval.bm25 import BM25Retriever


def span(a, b, doc='d'):
    return EvidenceSpan(document_id=doc, content_sha256='a'*64, canonical_sha256='b'*64, start_char=a, end_char=b)


def results(*spans):
    return [RetrievalResult(query_id='q', passage_id=str(i), document_id=s.document_id, rank=i, score=1, evidence_span=s, retriever_id='test') for i, s in enumerate(spans, 1)]


def test_union_coverage_mrr_and_deduplication():
    label = GoldLabel(query_id='q', answerable=True, evidence_spans=[span(0, 10), span(20, 30)])
    hits = results(span(0, 6), span(4, 10), span(0, 10), span(20, 30))
    assert score_at_k(label, hits, 1) == {'recall': 0, 'reciprocal_rank': 0}
    assert score_at_k(label, hits, 2) == {'recall': .5, 'reciprocal_rank': .5}
    assert score_at_k(label, hits, 3)['recall'] == .5
    assert score_at_k(label, hits, 4) == {'recall': 1, 'reciprocal_rank': .5}


def test_gaps_and_wrong_documents_fail():
    assert not fully_covered(span(0, 10), [span(0, 4), span(5, 10)])
    assert not fully_covered(span(0, 10), [span(0, 10, 'other')])
    assert fully_covered(span(0, 10), [span(0, 4), span(4, 10)])


def test_passage_labels():
    label = GoldLabel(query_id='q', answerable=True, relevant_passage_ids=['2', '3'], chunking_config_id='c')
    assert score_at_k(label, results(span(0, 1), span(1, 2)), 2) == {'recall': .5, 'reciprocal_rank': .5}


@pytest.mark.parametrize('data', [
    {'query_id': 'q'}, {'query_id': 'q', 'answerable': True},
    {'query_id': 'q', 'answerable': False, 'relevant_passage_ids': ['p'], 'chunking_config_id': 'c'},
    {'query_id': 'q', 'answerable': True, 'relevant_passage_ids': ['p']},
])
def test_malformed_labels(data):
    with pytest.raises(ValueError):
        GoldLabel.model_validate(data)


def test_missing_stale_and_unknown_labels(document, passages):
    query = Query(query_id='q', text='seedmarker')
    with pytest.raises(ValueError, match='Exactly one'):
        validate_labels([query], [], [document], passages)
    stale = passages[0].evidence_span.model_copy(update={'canonical_sha256': 'f'*64})
    with pytest.raises(ValueError, match='fingerprint'):
        validate_labels([query], [GoldLabel(query_id='q', answerable=True, evidence_spans=[stale])], [document], passages)
    with pytest.raises(ValueError, match='Unknown labeled passage'):
        validate_labels([query], [GoldLabel(query_id='q', answerable=True, relevant_passage_ids=['missing'], chunking_config_id='c')], [document], passages)


def test_unanswerable_exclusion_and_macro_average(document, passages):
    positive = next(p for p in passages if 'seedmarker' in p.text)
    queries = [Query(query_id='q', text='seedmarker'), Query(query_id='u', text='dragon')]
    gold = [GoldLabel(query_id='q', answerable=True, evidence_spans=[positive.evidence_span]), GoldLabel(query_id='u', answerable=False)]
    report = evaluate(BM25Retriever(), queries, gold, [document], passages, EvaluationConfig(ks=[1]))
    assert report['metrics'] == {'Recall@1': 1, 'cumulative_evidence_mrr_at_k': {'1': 1}, 'mrr_at_k': {'1': None}}
    assert report['total_queries'] == 2 and report['evaluated_queries'] == 1
    assert report['exclusions'] == [{'query_id': 'u', 'reason': 'explicitly_labeled_unanswerable'}]
    report = evaluate(BM25Retriever(), queries[1:], gold[1:], [document], passages, EvaluationConfig(ks=[1]))
    assert report['metrics'] == {'Recall@1': None, 'cumulative_evidence_mrr_at_k': {'1': None}, 'mrr_at_k': {'1': None}}


def test_interface_accepts_other_retriever(document, passages):
    class FixedRetriever:
        retriever_id = 'test-fixed-order'
        def index(self, items):
            self.items = items
        def search(self, query, k):
            return [RetrievalResult(query_id=query.query_id, passage_id=p.passage_id, document_id=p.document_id, rank=i, score=0, evidence_span=p.evidence_span, retriever_id=self.retriever_id) for i, p in enumerate(self.items[:k], 1)]
    q = Query(query_id='q', text='anything')
    gold = GoldLabel(query_id='q', answerable=True, evidence_spans=[passages[1].evidence_span])
    report = evaluate(FixedRetriever(), [q], [gold], [document], passages, EvaluationConfig(ks=[1, 2]))
    assert report['metrics']['cumulative_evidence_mrr_at_k']['2'] == .5
    assert report['metrics']['Recall@1'] == 0


def test_label_bounds_duplicates_and_chunk_config(document, passages):
    query = Query(query_id='q', text='seedmarker')
    oversized = passages[0].evidence_span.model_copy(update={'end_char': len(document.canonical_text) + 1})
    with pytest.raises(ValueError, match='outside'):
        validate_labels([query], [GoldLabel(query_id='q', answerable=True, evidence_spans=[oversized])], [document], passages)
    with pytest.raises(ValueError, match='Duplicate'):
        GoldLabel(query_id='q', answerable=True, evidence_spans=[passages[0].evidence_span]*2)
    with pytest.raises(ValueError, match='different chunking'):
        validate_labels([query], [GoldLabel(query_id='q', answerable=True, relevant_passage_ids=[passages[0].passage_id], chunking_config_id='wrong')], [document], passages)


def test_noncoverable_span_is_reported_not_excluded(document, passages):
    from krishisaarthi.preprocessing.chunking import DeterministicChunker
    from krishisaarthi.config import ChunkingConfig
    from krishisaarthi.ingestion.pdf import PDFIngestor
    from krishisaarthi.schemas import SourceMetadata
    from conftest import FIXTURES
    doc = PDFIngestor().ingest(FIXTURES / 'synthetic_wheat.pdf', SourceMetadata())
    chunks = DeterministicChunker(ChunkingConfig()).chunk(doc)
    gold_span = EvidenceSpan(document_id=doc.document_id, content_sha256=doc.content_sha256,
        canonical_sha256=doc.canonical_sha256, start_char=doc.pages[0].canonical_end - 2,
        end_char=doc.pages[1].canonical_start + 2)
    report = evaluate(BM25Retriever(), [Query(query_id='q', text='wheat')],
        [GoldLabel(query_id='q', answerable=True, evidence_spans=[gold_span])],
        [doc], chunks, EvaluationConfig(ks=[100]))
    assert report['metrics']['Recall@100'] == 0
    assert report['excluded_queries'] == 0
    assert report['warnings'][0]['reason'] == 'gold_span_not_coverable_by_corpus'


def test_false_string_is_not_unanswerable_label():
    with pytest.raises(ValueError):
        GoldLabel.model_validate({'query_id': 'q', 'answerable': 'false'})


def test_macro_average_includes_answerable_zero_hit(document, passages):
    class EmptyRetriever:
        retriever_id = 'test-empty'
        def index(self, items): pass
        def search(self, query, k): return []
    queries = [Query(query_id='q', text='x')]
    labels = [GoldLabel(query_id='q', answerable=True, evidence_spans=[passages[0].evidence_span])]
    report = evaluate(EmptyRetriever(), queries, labels, [document], passages, EvaluationConfig(ks=[1]))
    assert report['metrics'] == {'Recall@1': 0, 'cumulative_evidence_mrr_at_k': {'1': 0}, 'mrr_at_k': {'1': None}}
    assert report['evaluated_queries'] == 1


def test_tampered_passage_and_retriever_provenance(document, passages):
    corrupt = passages[0].model_copy(update={'text': 'tampered'})
    query = Query(query_id='q', text='x')
    gold = GoldLabel(query_id='q', answerable=True, evidence_spans=[passages[0].evidence_span])
    with pytest.raises(ValueError, match='canonical evidence'):
        validate_labels([query], [gold], [document], [corrupt, *passages[1:]])
    class WrongQueryRetriever:
        retriever_id = 'test-bad'
        def index(self, items): pass
        def search(self, query, k):
            p = passages[0]
            return [RetrievalResult(query_id='wrong', passage_id=p.passage_id, document_id=p.document_id,
                rank=1, score=0, evidence_span=p.evidence_span, retriever_id=self.retriever_id)]
    with pytest.raises(ValueError, match='inconsistent'):
        evaluate(WrongQueryRetriever(), [query], [gold], [document], passages, EvaluationConfig(ks=[1]))


@pytest.mark.parametrize('mode', ['evidence_spans', 'passage_ids', 'mixed'])
def test_report_names_distinguish_relevance_policies(document, passages, mode):
    class FixedRetriever:
        retriever_id = 'test-fixed'
        def index(self, items):
            self.items = items
        def search(self, query, k):
            return [RetrievalResult(query_id=query.query_id, passage_id=p.passage_id,
                document_id=p.document_id, rank=i, score=0, evidence_span=p.evidence_span,
                retriever_id=self.retriever_id) for i, p in enumerate(self.items[:k], 1)]
    labels = []
    if mode in ('evidence_spans', 'mixed'):
        labels.append(GoldLabel(query_id='span', answerable=True, evidence_spans=[passages[1].evidence_span]))
    if mode in ('passage_ids', 'mixed'):
        labels.append(GoldLabel(query_id='passage', answerable=True,
            relevant_passage_ids=[passages[0].passage_id], chunking_config_id=passages[0].chunking_config_id))
    report = evaluate(FixedRetriever(), [Query(query_id=l.query_id, text='fixture') for l in labels],
        labels, [document], passages, EvaluationConfig(ks=[1, 2]))
    assert report['metrics']['cumulative_evidence_mrr_at_k'] == (
        {'1': 0, '2': .5} if mode != 'passage_ids' else {'1': None, '2': None})
    assert report['metrics']['mrr_at_k'] == (
        {'1': 1, '2': 1} if mode != 'evidence_spans' else {'1': None, '2': None})
    assert not any(key.startswith('MRR@') for key in report['metrics'])
