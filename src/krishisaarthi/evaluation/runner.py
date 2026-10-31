from .labels import validate_labels
from .metrics import score_at_k, complete_evidence_coverage_at_k
from ..config import fingerprint


def evaluate(retriever, queries, gold, documents, passages, config):
    """Index once; evaluation never depends on BM25 or a model tokenizer."""
    labels, indexed, warnings = validate_labels(queries, gold, documents, passages)
    retriever.index(passages)
    rows, excluded, diagnostics = [], [], []
    limit = max(config.ks)
    for query in queries:
        label = labels[query.query_id]
        diagnostic = getattr(retriever, "diagnostics", None)
        if callable(diagnostic):
            diagnostics.append(diagnostic(query, limit))
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
        rows.append({"query_id": query.query_id, "language": query.language or "unspecified",
            "relevance_type": "evidence_spans" if label.evidence_spans else "passage_ids",
            "metrics": {str(k): {**score_at_k(label, results, k),
                "complete_evidence_coverage": complete_evidence_coverage_at_k(label, results, k) if label.evidence_spans else None}
                for k in config.ks},
            "results": [r.model_dump() for r in results]})
    metrics = aggregate_metrics(rows, config.ks)
    languages = ["en", "hi", "hinglish"] + (["unspecified"] if any(q.language is None for q in queries) else [])
    per_language = {}
    for language in languages:
        subset = [r for r in rows if r["language"] == language]
        total = sum((q.language or "unspecified") == language for q in queries)
        per_language[language] = {
            "total_queries": total, "evaluated_queries": len(subset),
            "excluded_queries": total - len(subset),
            "span_labeled_queries": sum(r["relevance_type"] == "evidence_spans" for r in subset),
            "metrics": aggregate_metrics(subset, config.ks)}
    return {"schema_version": 2, "coverage_rule": config.coverage_rule,
        "coverage_description": "100% of every gold character interval must be covered by the union of same-document, same-version retrieved chunks; each gold span counted once. cumulative_evidence_mrr_at_k uses the earliest prefix fully covering any gold span; mrr_at_k uses the first individually judged relevant passage.",
        "aggregation": "Macro average over answerable queries; each policy-specific metric uses only its relevance type. Complete evidence coverage requires ALL spans. Null for empty denominators. Overall remains query-macro, not a new group weighting.",
        "per_language": per_language,
        "complete_coverage_description": "1 iff every required gold evidence span is fully covered at K; strict same-document/version interval union. Unanswerable and passage-relevance labels excluded.",
        "diagnostics": {"scope": "All queries including explicitly unanswerable; corpus-wide no-positive-score flag, retrieved zero scores at max K. Scores are not relevance judgments; ranking and metric inputs are unchanged.",
            "available": bool(diagnostics), "overall": diagnostic_summary(diagnostics),
            "per_language": {lang: diagnostic_summary([d for d in diagnostics if d["language"] == lang]) for lang in languages},
            "per_query": diagnostics},
        "total_queries": len(queries), "evaluated_queries": len(rows), "excluded_queries": len(excluded),
        "exclusions": excluded, "warnings": warnings, "metrics": metrics, "per_query": rows,
        "retriever_id": retriever.retriever_id,
        "query_fingerprint": fingerprint([q.model_dump() for q in queries]),
        "label_fingerprint": fingerprint([l.model_dump() for l in gold])}


def aggregate_metrics(rows, ks):
    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = sum(r["metrics"][str(k)]["recall"] for r in rows) / len(rows) if rows else None
    # Keep the two relevance policies explicitly named; never label a mixed
    # reciprocal-rank average as either cumulative evidence or passage MRR.
    for name, relevance_type in (("cumulative_evidence_mrr_at_k", "evidence_spans"),
                                 ("mrr_at_k", "passage_ids")):
        matching = [row for row in rows if row["relevance_type"] == relevance_type]
        metrics[name] = {
            str(k): sum(row["metrics"][str(k)]["reciprocal_rank"] for row in matching) / len(matching) if matching else None
            for k in ks
        }
    spans = [r for r in rows if r["relevance_type"] == "evidence_spans"]
    metrics["complete_evidence_coverage_at_k"] = {
        str(k): sum(r["metrics"][str(k)]["complete_evidence_coverage"] for r in spans) / len(spans) if spans else None
        for k in ks}
    return metrics


def diagnostic_summary(rows):
    count = sum(row["no_positive_match"] for row in rows)
    return {"queries": len(rows), "no_positive_match_count": count,
            "no_positive_match_rate": count / len(rows) if rows else None,
            "no_positive_match_query_ids": [r["query_id"] for r in rows if r["no_positive_match"]],
            "zero_score_result_count": sum(r["zero_score_result_count"] for r in rows)}
