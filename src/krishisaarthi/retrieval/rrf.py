"""Equal-weight reciprocal rank fusion; input scores only control optional gating."""
import math
from ..config import fingerprint
from ..schemas import RetrievalResult


class RRFRetriever:
    def __init__(self, lexical, dense, constant=60, gate_nonpositive_lexical=False,
                 require_complete=False):
        if isinstance(constant, bool) or not math.isfinite(constant) or constant <= 0:
            raise ValueError("RRF constant must be finite and positive")
        self.lexical, self.dense = lexical, dense
        self.constant = constant
        self.gate_nonpositive_lexical = gate_nonpositive_lexical
        self.require_complete = require_complete
        self.retriever_id = 'rrf-' + fingerprint({
            'inputs': [lexical.retriever_id, dense.retriever_id], 'constant': constant,
            'gate_nonpositive_lexical': gate_nonpositive_lexical,
            'require_complete': require_complete, 'weights': [1, 1],
            'tie_break': 'passage_id', 'gated_ranks': 'original'})
        self._passages = None

    def index(self, passages):
        if not passages or len({p.passage_id for p in passages}) != len(passages):
            raise ValueError('Nonempty corpus with unique passage IDs required')
        self._passages = {p.passage_id: p for p in passages}
        self.lexical.index(passages)
        self.dense.index(passages)

    def search(self, query, k):
        if k <= 0 or self._passages is None:
            raise ValueError('Index first and use positive K')
        scores = {}
        for number, retriever in enumerate((self.lexical, self.dense)):
            results = list(retriever.search(query, len(self._passages)))
            seen = set()
            for rank, result in enumerate(results, 1):
                if result.passage_id in seen:
                    raise ValueError('Duplicate passage in an input ranking')
                seen.add(result.passage_id)
                passage = self._passages.get(result.passage_id)
                if (passage is None or result.query_id != query.query_id or result.rank != rank
                        or result.document_id != passage.document_id
                        or result.evidence_span != passage.evidence_span
                        or not math.isfinite(result.score)):
                    raise ValueError('Invalid input ranking or provenance')
                if number == 0 and self.gate_nonpositive_lexical and result.score <= 0:
                    continue
                scores[result.passage_id] = scores.get(result.passage_id, 0.) + 1 / (self.constant + rank)
            if self.require_complete and seen != self._passages.keys():
                raise ValueError('Complete corpus ranking required')
        ordered = sorted(scores, key=lambda pid: (-scores[pid], pid))[:k]
        return [RetrievalResult(query_id=query.query_id, passage_id=pid,
                document_id=self._passages[pid].document_id, rank=rank, score=scores[pid],
                evidence_span=self._passages[pid].evidence_span, retriever_id=self.retriever_id)
                for rank, pid in enumerate(ordered, 1)]
