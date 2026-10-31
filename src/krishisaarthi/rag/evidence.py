from datetime import datetime
from ..schemas import Query
from ..config import text_hash
from ..validation import require
from .models import EvidencePack, Sufficiency

class EvidenceSelector:
    """Retrieval adapter; documents/passages must be loaded through corpus validation."""
    def __init__(self, retriever, documents, passages):
        # Source-byte validation is reused; no queries, labels or sign-off files read.
        from ..evaluation.labels import validate_corpus
        self.documents, self.passages = validate_corpus(documents, passages)
        self.retriever = retriever
        retriever.index(passages)

    def select(self, question: Query, k=5):
        require(k > 0, 'K must be positive')
        hits=list(self.retriever.search(question,k));seen=set();chosen=[]
        require(len(hits)<=k,'Excess retrieval results')
        for rank,h in enumerate(hits,1):
            p=self.passages.get(h.passage_id)
            require(p is not None and h.passage_id not in seen,'Unknown or duplicate passage')
            require(h.query_id==question.query_id and h.rank==rank and
                    h.document_id==p.document_id and h.evidence_span==p.evidence_span,
                    'Retrieved provenance mismatch')
            seen.add(h.passage_id);chosen.append(p.model_copy(deep=True))
        return EvidencePack(question=question,passages=chosen,retriever_id=self.retriever.retriever_id,
            document_metadata={p.document_id:self.documents[p.document_id].domain_metadata for p in chosen})

class CitationValidator:
    """Structural source binding only. Exact quotes do NOT prove claim entailment."""
    def validate(self, claims, evidence):
        passages={p.passage_id:p for p in evidence.passages};errors=[]
        for claim in claims:
            if not claim.citations:errors.append('Claim lacks citations')
            for c in claim.citations:
                p=passages.get(c.passage_id)
                if p is None:
                    errors.append('Citation references unprovided passage');continue
                s=p.evidence_span
                if not s.start_char<=c.start_char<c.end_char<=s.end_char:
                    errors.append('Citation offsets outside provided passage');continue
                if p.text[c.start_char-s.start_char:c.end_char-s.start_char]!=c.quote:
                    errors.append('Citation quote does not match source interval')
        return errors

class EvidenceSufficiencyChecker:
    """Conservative default: no autonomous semantic sufficiency claim.

    A fresh, bound support review permits an attempt, not certification of output.
    Missing review or ANY unresolved component means insufficient evidence.
    """
    def assess(self, evidence, review=None):
        if not evidence.passages:return Sufficiency(sufficient_to_attempt=False,reasons=['No retrieved evidence'])
        if review is None:return Sufficiency(sufficient_to_attempt=False,reasons=['Semantic sufficiency not established; request-specific evidence review required'])
        reasons=[]
        if review.evidence_fingerprint!=evidence.fingerprint:reasons.append('Stale review: question/evidence fingerprint differs')
        try:
            dt=datetime.fromisoformat(review.reviewed_at)
            if dt.tzinfo is None:raise ValueError()
        except ValueError:reasons.append('Review timestamp requires timezone')
        if not review.reviewer_id.strip():reasons.append('Missing reviewer identity')
        if not review.all_question_components_covered:reasons.append('Not all question components supported')
        if not review.applicability_checked:reasons.append('Applicability not reviewed')
        reasons.extend(review.unresolved_concerns)
        from .models import Claim
        for component in review.components:
            reasons.extend(CitationValidator().validate([Claim(text=component.component,citations=component.support)],evidence))
        return Sufficiency(sufficient_to_attempt=not reasons,reasons=reasons or ['Bound human support review permits a draft attempt; no factual certification'])
