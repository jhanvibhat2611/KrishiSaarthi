from .labels import validate_labels
from .metrics import score_at_k
from ..config import fingerprint


def evaluate(retriever, queries, gold, documents, passages, config):
    """Index once; evaluation never depends on BM25 or a model tokenizer."""
    labels, indexed, warnings = validate_labels(queries, gold, documents, passages)
    retriever.index(passages)
    rows, excluded = [], []
    limit = max(config.ks)
    for query in queries:
        label = labels[query.query_id]
        if not label.answerable:
            excluded.append({"query_id": query.query_id, "reason": "explicitly_labeled_unanswerable"})
            continue
        results = list(retriever.search(query, limit))
        if len(results) > limit or len({r.passage_id for r in results}) != len(results):
            raise ValueError("Retriever returned duplicate or excess results")
        for rank, result in enumerate(results, 1):
            if result.passage_id not in indexed:
                raise ValueError("Retriever returned an unknown passage")
            passage = indexed[result.passage_id]
            if (result.query_id != query.query_id or result.rank != rank or
                result.document_id != passage.document_id or result.evidence_span != passage.evidence_span):
                raise ValueError("Retriever returned inconsistent rank, query, or provenance")
        rows.append({"query_id": query.query_id,
            "relevance_type": "evidence_spans" if label.evidence_spans else "passage_ids",
            "metrics": {str(k): score_at_k(label, results, k) for k in config.ks},
            "results": [r.model_dump() for r in results]})
    metrics = {}
    for k in config.ks:
        metrics[f"Recall@{k}"] = sum(r["metrics"][str(k)]["recall"] for r in rows) / len(rows) if rows else None
    # Keep the two relevance policies explicitly named; never label a mixed
    # reciprocal-rank average as either cumulative evidence or passage MRR.
    for name, relevance_type in (("cumulative_evidence_mrr_at_k", "evidence_spans"),
                                 ("mrr_at_k", "passage_ids")):
        matching = [row for row in rows if row["relevance_type"] == relevance_type]
        metrics[name] = {
            str(k): sum(row["metrics"][str(k)]["reciprocal_rank"] for row in matching) / len(matching) if matching else None
            for k in config.ks
        }
    return {"schema_version": 1, "coverage_rule": config.coverage_rule,
        "coverage_description": "100% of every gold character interval must be covered by the union of same-document, same-version retrieved chunks; each gold span counted once. cumulative_evidence_mrr_at_k uses the earliest prefix fully covering any gold span; mrr_at_k uses the first individually judged relevant passage.",
        "aggregation": "Recall: macro average over answerable queries. Each named reciprocal-rank metric: macro average over answerable queries using its relevance type; null if none",
        "total_queries": len(queries), "evaluated_queries": len(rows), "excluded_queries": len(excluded),
        "exclusions": excluded, "warnings": warnings, "metrics": metrics, "per_query": rows,
        "retriever_id": retriever.retriever_id,
        "query_fingerprint": fingerprint([q.model_dump() for q in queries]),
        "label_fingerprint": fingerprint([l.model_dump() for l in gold])}
