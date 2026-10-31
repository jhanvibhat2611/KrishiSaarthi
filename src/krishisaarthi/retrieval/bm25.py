from rank_bm25 import BM25Okapi
from ..config import BM25Config, fingerprint
from ..preprocessing.tokenization import UnicodeTokenizer, TOKENIZER_ID
from ..schemas import RetrievalResult


class BM25Retriever:
    def __init__(self, config: BM25Config | None = None):
        self.config = config or BM25Config()
        self.retriever_id = "bm25-okapi-" + fingerprint({"config": self.config.model_dump(), "tokenizer": TOKENIZER_ID})
        self.tokenizer = UnicodeTokenizer()
        self._model = None
        self._passages = []

    def index(self, passages):
        if not passages:
            raise ValueError("Cannot index an empty corpus")
        if len({p.passage_id for p in passages}) != len(passages):
            raise ValueError("Duplicate passage IDs")
        ordered = sorted(passages, key=lambda p: p.passage_id)
        terms = [self.tokenizer.terms(p.text) for p in ordered]
        if any(not row for row in terms):
            raise ValueError("Passages must have tokens")
        self._model = BM25Okapi(terms, k1=self.config.k1, b=self.config.b, epsilon=self.config.epsilon)
        self._passages = ordered

    def search(self, query, k):
        if k <= 0:
            raise ValueError("k must be positive")
        if self._model is None:
            raise ValueError("Index the corpus before searching")
        scores = self._model.get_scores(self.tokenizer.terms(query.text))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self._passages[i].passage_id))[:k]
        return [RetrievalResult(query_id=query.query_id, passage_id=self._passages[i].passage_id,
            document_id=self._passages[i].document_id, rank=rank, score=float(scores[i]),
            evidence_span=self._passages[i].evidence_span, retriever_id=self.retriever_id)
            for rank, i in enumerate(order, 1)]


    def diagnostics(self, query, k):
        """Corpus-wide lexical/score diagnostics; never filter or rerank results."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._model is None:
            raise ValueError("Index the corpus before diagnostics")
        terms = self.tokenizer.terms(query.text)
        scores = self._model.get_scores(terms)
        positive = sum(float(score) > 0 for score in scores)
        overlap = len(set(terms).intersection(self._model.idf))
        results = self.search(query, k)
        return {
            "query_id": query.query_id, "language": query.language or "unspecified",
            "query_token_count": len(terms), "vocabulary_overlap_count": overlap,
            "empty_token_query": not terms, "lexical_no_match": overlap == 0,
            "corpus_positive_score_count": positive,
            "no_positive_match": positive == 0,
            "all_corpus_scores_zero": all(float(score) == 0 for score in scores),
            "retrieved_count": len(results),
            "positive_score_result_count": sum(r.score > 0 for r in results),
            "negative_score_result_count": sum(r.score < 0 for r in results),
            "zero_score_result_count": sum(r.score == 0 for r in results),
            "zero_score_passage_ids": [r.passage_id for r in results if r.score == 0],
        }
