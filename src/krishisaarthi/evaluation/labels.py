import hashlib
from pathlib import Path
from ..config import text_hash
from .metrics import fully_covered


def unique_by(records, field):
    mapping = {getattr(r, field): r for r in records}
    if len(mapping) != len(records):
        raise ValueError(f"Duplicate {field}")
    return mapping


def validate_span(span, documents):
    doc = documents.get(span.document_id)
    if doc is None:
        raise ValueError(f"Unknown document: {span.document_id}")
    if span.content_sha256 != doc.content_sha256 or span.canonical_sha256 != doc.canonical_sha256:
        raise ValueError("Evidence fingerprint does not match the document version")
    if not 0 <= span.start_char < span.end_char <= len(doc.canonical_text):
        raise ValueError("Evidence offsets are outside canonical text")


def validate_page_file(page, document):
    """Declared storage must exist and equal the exact canonical page bytes."""
    if page.extracted_text_path is None:
        return  # In-memory ingestion has no declared extracted file.
    try:
        payload = Path(page.extracted_text_path).read_bytes()
    except OSError as exc:
        raise ValueError("Extracted page file is missing or unreadable") from exc
    expected = document.canonical_text[page.canonical_start:page.canonical_end].encode("utf-8")
    if payload != expected:
        raise ValueError("Extracted page file does not match canonical page content")


def page_identity(page):
    return (page.page_number, page.canonical_start, page.canonical_end)


def validate_corpus(documents, passages):
    docs = unique_by(documents, "document_id")
    indexed = unique_by(passages, "passage_id")
    if not docs or not indexed:
        raise ValueError("Documents and passages must be nonempty")
    for doc in documents:
        if text_hash(doc.canonical_text) != doc.canonical_sha256:
            raise ValueError("Canonical text fingerprint mismatch")
        try:
            source_hash = hashlib.sha256(Path(doc.original_file_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError("Original source file is missing or unreadable") from exc
        if source_hash != doc.content_sha256:
            raise ValueError("Original source fingerprint mismatch")
        last = 0
        for page in doc.pages:
            if not 0 <= page.canonical_start <= page.canonical_end <= len(doc.canonical_text) or page.canonical_start < last:
                raise ValueError("Invalid page offsets")
            validate_page_file(page, doc)
            last = page.canonical_end
    for passage in passages:
        span = passage.evidence_span
        validate_span(span, docs)
        doc = docs[span.document_id]
        if passage.document_id != span.document_id or passage.text != doc.canonical_text[span.start_char:span.end_char]:
            raise ValueError("Passage does not match its canonical evidence span")
        expected = [p for p in doc.pages if p.canonical_start < span.end_char and p.canonical_end > span.start_char]
        if [page_identity(p) for p in passage.page_references] != [page_identity(p) for p in expected]:
            raise ValueError("Passage page references do not match document")
        for actual, source in zip(passage.page_references, expected):
            if actual.extracted_text_path != source.extracted_text_path and (
                actual.extracted_text_path is None or source.extracted_text_path is None
            ):
                raise ValueError("Different page paths require two verifiable files")
            validate_page_file(actual, doc)
            # The document file was also checked above. Different paths are accepted
            # only when both match the same exact canonical page bytes. The parent
            # evidence span has already bound document ID and both source hashes.
    return docs, indexed


def validate_labels(queries, labels, documents, passages):
    docs, indexed = validate_corpus(documents, passages)
    query_map = unique_by(queries, "query_id")
    gold = unique_by(labels, "query_id")
    if not query_map:
        raise ValueError("Evaluation queries must be nonempty")
    if query_map.keys() != gold.keys():
        raise ValueError("Exactly one explicit gold label is required for every query; no extra labels")
    warnings = []
    for label in labels:
        for span in label.evidence_spans:
            validate_span(span, docs)
            if not fully_covered(span, [p.evidence_span for p in passages]):
                warnings.append({"query_id": label.query_id, "reason": "gold_span_not_coverable_by_corpus", "span": span.model_dump()})
        for pid in label.relevant_passage_ids:
            if pid not in indexed:
                raise ValueError(f"Unknown labeled passage: {pid}")
            if indexed[pid].chunking_config_id != label.chunking_config_id:
                raise ValueError("Passage labels refer to a different chunking configuration")
    return gold, indexed, warnings
