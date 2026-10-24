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
