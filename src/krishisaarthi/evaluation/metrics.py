"""Strict interval-union coverage; whitespace is significant."""


def span_key(span):
    return (span.document_id, span.content_sha256, span.canonical_sha256)


def fully_covered(gold, retrieved_spans):
    cursor = gold.start_char
    intervals = sorted((s.start_char, s.end_char) for s in retrieved_spans if span_key(s) == span_key(gold))
    for start, end in intervals:
        if end <= cursor:
            continue
        if start > cursor:
            break
        cursor = max(cursor, end)
        if cursor >= gold.end_char:
            return True
    return False


def score_at_k(label, results, k):
    """First hit = earliest prefix completing ANY positive evidence unit."""
    if not label.answerable:
        raise ValueError("Unanswerable queries are excluded, not scored")
    retrieved_spans, ids = [], set()
    first_hit, covered = None, 0
    total = len(label.evidence_spans) or len(label.relevant_passage_ids)
    for rank, result in enumerate(results[:k], 1):
        retrieved_spans.append(result.evidence_span)
        ids.add(result.passage_id)
        if label.evidence_spans:
            covered = sum(fully_covered(span, retrieved_spans) for span in label.evidence_spans)
        else:
            covered = len(ids.intersection(label.relevant_passage_ids))
        if covered and first_hit is None:
            first_hit = rank
    return {"recall": covered / total, "reciprocal_rank": 1 / first_hit if first_hit else 0.0}
