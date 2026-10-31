"""Fail-closed admission of manually approved source regions."""
import hashlib
from pathlib import Path
import re
from ..config import fingerprint, text_hash
from ..schemas import PageReference
from ..evaluation.labels import unique_by, validate_span, validate_corpus
from ..preprocessing.chunking import DeterministicChunker
from .models import ReviewedRegion


def review_fingerprint(region):
    """Approval binds offsets, boundaries and printed labels; changing them invalidates it."""
    return fingerprint({'span': region.evidence_span.model_dump(), 'physical_page': region.physical_page,
        'printed_page_label': region.printed_page_label, 'section_boundaries': region.section_boundaries,
        'policy': 'reviewed-english-wheat-privacy-v1'})


def validate_regions(documents, regions, issues):
    docs = unique_by(documents, 'document_id')
    unique_by(regions, 'region_id')
    for doc in documents:
        if text_hash(doc.canonical_text) != doc.canonical_sha256:
            raise ValueError('Canonical source text was modified')
        if hashlib.sha256(Path(doc.original_file_path).read_bytes()).hexdigest() != doc.content_sha256:
            raise ValueError('Original document was modified')
    approved = []
    for raw in regions:
        # Revalidate model_copy or programmatically constructed records too.
        region = ReviewedRegion.model_validate(raw.model_dump())
        span = region.evidence_span
        validate_span(span, docs)
        doc = docs[span.document_id]
        matching = [p for p in doc.pages if p.page_number == region.physical_page]
        if len(matching) != 1:
            raise ValueError('Physical page is not a unique source page')
        page = matching[0]
        if not page.canonical_start <= span.start_char < span.end_char <= page.canonical_end:
            raise ValueError('Region must remain within its physical page')
        if region.status != 'approved':
            continue
        if region.reviewed_content_fingerprint != review_fingerprint(region):
            raise ValueError('Approval fingerprint is stale; inspect and review the changed region')
        relevant = [i for i in issues if i.document_id == doc.document_id
                    and i.start_char < span.end_char and i.end_char > span.start_char]
        # Derive intersections from trusted audit, not user-editable issue_ids.
        if any(not region.resolutions.get(i.issue_id, '').strip() for i in relevant):
            raise ValueError('Unresolved audit issues overlap an approved region')
        text = doc.canonical_text[span.start_char:span.end_char]
        if re.search(r'[\u0900-\u097f]', text):
            raise ValueError('Approved English region contains Devanagari')
        if not text.strip():
            raise ValueError('Cannot approve whitespace-only evidence')
        approved.append(region)
    approved.sort(key=lambda r: (r.evidence_span.document_id, r.evidence_span.start_char))
    for left, right in zip(approved, approved[1:]):
        if (left.evidence_span.document_id == right.evidence_span.document_id and
                left.evidence_span.end_char > right.evidence_span.start_char):
            raise ValueError('Approved regions must not overlap')
    return approved


def build_passages(documents, regions, issues, config):
    approved = validate_regions(documents, regions, issues)
    if not approved:
        raise ValueError('No manually approved regions; research index was not created')
    selection_hash = fingerprint([r.model_dump() for r in approved])
    return _chunk_regions(documents, approved, config, selection_hash)


def _chunk_regions(documents, approved, config, selection_hash, selection_metadata=None):
    """Shared chunk construction; callers must first validate their admission gate."""
    base = DeterministicChunker(config)
    config_id = fingerprint({'base_chunking_config_id': base.config_id,
                             'selection_fingerprint': selection_hash,
                             'boundary_policy': 'reviewed-page-regions-v1'})
    docs = {d.document_id: d for d in documents}
    passages = []
    for region in approved:
        doc = docs[region.evidence_span.document_id]
        source_page = next(p for p in doc.pages if p.page_number == region.physical_page)
        points = [region.evidence_span.start_char, *region.section_boundaries, region.evidence_span.end_char]
        # Feed absolute-offset windows to the existing chunker. Canonical text is
        # untouched; chunks get the original page reference back before validation.
        windows = [PageReference(page_number=region.physical_page, canonical_start=a, canonical_end=b)
                   for a, b in zip(points, points[1:])]
        view = doc.model_copy(update={'pages': windows})
        for passage in base.chunk(view):
            passage.page_references = [source_page]
            passage.chunking_config_id = config_id
            passage.passage_id = 'passage-' + fingerprint({'span': passage.evidence_span.model_dump(), 'chunking': config_id})
            passage.domain_metadata = {**passage.domain_metadata,
                'region_id': region.region_id, 'selection_fingerprint': selection_hash,
                'printed_page_label': region.printed_page_label, 'reviewer': region.reviewer,
                'reviewed_at': region.reviewed_at, 'approval': 'manual', 'publication_allowed': False}
            if selection_metadata is not None:
                passage.domain_metadata.update(selection_metadata[region.region_id])
                passage.domain_metadata.pop('reviewer', None)
                passage.domain_metadata.pop('reviewed_at', None)
            passages.append(passage)
    validate_corpus(documents, passages)
    return passages, selection_hash
