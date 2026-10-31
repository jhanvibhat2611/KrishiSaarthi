"""Deterministic diagnostics, not OCR, language certification or human approval."""
import hashlib
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import pypdf
from ..config import fingerprint
from ..ingestion.storage import make_document
from ..schemas import SourceMetadata, EvidenceSpan
from .models import Issue, ReviewedRegion

AUDIT_VERSION = 'pilot-audit-v2'
SECTION_NAMES = {'Summary', 'Haryana', 'Punjab', 'Himachal Pradesh', 'Jammu & Kashmir',
                 'Jammu& Kashmir', 'Uttar Pradesh', 'Uttarakhand', 'Madhya Pradesh',
                 'Rajasthan', 'Gujarat', 'Bihar', 'West Bengal', 'Karnataka', 'Maharashtra'}


def extract(entry, path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry.content_sha256:
        raise ValueError('Registry file hash differs from original')
    reader = pypdf.PdfReader(path)
    if reader.is_encrypted:
        raise ValueError('Encrypted PDF is unsupported')
    pages, diagnostics = [], []
    for number, page in enumerate(reader.pages, 1):
        error, text = None, ''
        try:
            text = page.extract_text() or ''
        except Exception as exc:
            error = type(exc).__name__  # No potentially identifying exception payloads.
        try:
            image_count = len(page.images)
        except Exception:
            image_count = None
        pages.append((number, text))
        diagnostics.append({'physical_page': number, 'extraction_error': error,
                            'image_count': image_count})
    metadata = SourceMetadata(document_id=entry.document_id, source_url=entry.source_url,
        domain_metadata={'crop': 'wheat', 'country': 'India', 'corpus_scope': 'pilot_only',
                         'source_languages': entry.source_languages, 'license_status': entry.license_status,
                         'publication_date': entry.publication_date, 'date_precision': entry.date_precision,
                         'publisher': entry.publisher, 'geographic_applicability': entry.geographic_applicability,
                         'copyright_restrictions': entry.copyright_restrictions})
    doc = make_document(path, metadata, pages, 'application/pdf', 'pypdf', pypdf.__version__)
    doc.extraction_metadata['audit_version'] = AUDIT_VERSION
    doc.extraction_metadata['empty_pages'] = [n for n, t in pages if not t.strip()]
    return doc, diagnostics


def blocks(text, offset):
    """Candidate boundaries only; never alter source text or approve sections."""
    points = {0, len(text)}
    for match in re.finditer(r'\n[ \t]*\n', text):
        points.add(match.end())
    for match in re.finditer(r'^.*$', text, re.M):
        if match.group().strip() in SECTION_NAMES:
            points.add(match.start())
    ordered = sorted(points)
    for a, b in zip(ordered, ordered[1:]):
        part = text[a:b]
        if part.strip():
            yield offset + a, offset + b, part


def audit(documents, entries, diagnostics):
    issues, pages, paragraph_refs = [], [], []
    entry_map = {e.document_id: e for e in entries}

    def flag(doc, page, code, start, end, detail):
        key = {'document': doc.document_id, 'canonical': doc.canonical_sha256,
               'page': page.page_number, 'code': code, 'start': start, 'end': end}
        item = Issue(issue_id='issue-' + fingerprint(key)[:20], code=code,
            document_id=doc.document_id, physical_page=page.page_number,
            start_char=start, end_char=end, detail=detail)
        if item.issue_id not in {i.issue_id for i in issues}:
            issues.append(item)

    for doc in documents:
        entry = entry_map[doc.document_id]
        for page, diagnostic in zip(doc.pages, diagnostics[doc.document_id], strict=True):
            a, b = page.canonical_start, page.canonical_end
            text = doc.canonical_text[a:b]
            devanagari = len(re.findall(r'[\u0900-\u097f]', text))
            page_row = {**diagnostic, 'document_id': doc.document_id,
                'printed_page_label': entry.printed_page_labels.get(str(page.page_number)),
                'printed_label_status': 'observed_unverified', 'canonical_start': a, 'canonical_end': b,
                'characters': len(text), 'devanagari_characters': devanagari,
                'replacement_characters': text.count('\ufffd'), 'status': 'pending_review'}
            pages.append(page_row)
            if diagnostic['extraction_error']:
                flag(doc, page, 'extraction_error', a, b, 'Page extraction failed; original retained.')
            if not text.strip():
                flag(doc, page, 'no_text_scan_candidate', a, b, 'No text: distinguish blank, scanned or failed page manually; no OCR applied.')
            elif len(text.strip()) < 100:
                flag(doc, page, 'sparse_text', a, b, 'Sparse text; inspect image-only information and layout.')
            if '\ufffd' in text:
                flag(doc, page, 'replacement_characters', a, b, 'Text contains replacement characters.')
            if re.search(r'[\x00-\x08\x0b\x0e-\x1f\x7f]', text):
                flag(doc, page, 'control_characters', a, b, 'Unexpected control characters; glyph extraction requires visual verification. Canonical text is unchanged.')
            for match in re.finditer(r'^.*Wheat Crop Health Newsletter.*$', text, re.M | re.I):
                flag(doc, page, 'running_header', a+match.start(), a+match.end(), 'Repeated publication header/footer; not agricultural evidence.')
            if page.page_number == 1:
                flag(doc, page, 'cover', a, b, 'Cover: retain provenance, exclude from evidence index.')
            if re.search(r'Issued by:|Compiled and Edited by:', text, re.I):
                flag(doc, page, 'credits', a, b, 'Publication credits/contact details; not evidence.')
            # Tables require row/column verification. Bound from title to next known section.
            table = re.search(r'Table\s*\d|S\. No\.', text)
            if table:
                next_section = next((m.start() for m in re.finditer(r'^.*$', text, re.M)
                    if m.start() > table.start() and m.group().strip() in SECTION_NAMES), len(text))
                flag(doc, page, 'table_alignment', a+table.start(), a+next_section, 'Table reading order and missing cells require visual verification.')
            for start, end, part in blocks(text, a):
                if re.search(r'[\u0900-\u097f]', part):
                    flag(doc, page, 'non_english', start, end, 'Contains Devanagari; select English evidence only.')
                if re.search(r'\bbarley\b', part, re.I):
                    flag(doc, page, 'crop_scope', start, end, 'Mixed crop content; isolate wheat evidence manually.')
                if re.search(r'Name and Address|Name.*farmer|\bS/o\b|\bW/o\b|\bD/o\b|\b(?:Dr\.?|Mr\.?|Mrs\.?|Shri|Smt)\s|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', part, re.I):
                    flag(doc, page, 'privacy', start, end, 'Potential personal or contact information; exclude unnecessary identifying spans.')
                if re.search(r'\b(?:whe\s+at|healt\s+h|fie\s+lds|surve\s+y|observ\s+ed)\b', part, re.I):
                    flag(doc, page, 'split_word_candidate', start, end, 'Possible PDF word fragmentation; inspect against original without automatic repair.')
                if part.strip() in SECTION_NAMES:
                    flag(doc, page, 'heading_only', start, end, 'Standalone heading; manually combine with contiguous reviewed evidence or exclude.')
                if re.search(r'No report received', part, re.I):
                    flag(doc, page, 'missing_report', start, end, 'Missing survey information is not evidence of absence of disease.')
                normalized = re.sub(r'\s+', ' ', part).casefold().strip()
                if len(normalized) >= 180:
                    paragraph_refs.append((doc, page, start, end, normalized))
            for anchor in entry.known_issue_anchors:
                if anchor['physical_page'] != page.page_number:
                    continue
                match = text.find(str(anchor['start_text']))
                if match < 0:
                    flag(doc, page, 'anchor_not_found', a, b, 'Previously inspected issue anchor no longer matches; re-review extraction.')
                    continue
                stop_text = anchor.get('end_before')
                stop = text.find(str(stop_text), match + 1) if stop_text else len(text)
                if stop < 0:
                    flag(doc, page, 'anchor_not_found', a, b, 'Issue end anchor not found; re-review extraction.')
                    continue
                flag(doc, page, str(anchor['code']), a+match, a+stop, str(anchor['detail']))
    # Comparison normalization is diagnostic only; canonical bytes remain unchanged.
    for index, first in enumerate(paragraph_refs):
        for second in paragraph_refs[index+1:]:
            x, y = first[4], second[4]
            if min(len(x), len(y)) / max(len(x), len(y)) < .65:
                continue
            score = SequenceMatcher(None, x, y, autojunk=False).ratio()
            if score >= .85:
                for current, other in ((first, second), (second, first)):
                    doc, page, start, end, _ = current
                    flag(doc, page, 'duplicate_candidate', start, end,
                         f'Near-duplicate text candidate (similarity {score:.3f}) with {other[0].document_id} page {other[1].page_number}, offsets [{other[2]},{other[3]}); human comparison required.')
    # Exact full document duplicates remain separate provenance records, but are held.
    hashes = defaultdict(list)
    for doc in documents:
        hashes[doc.content_sha256].append(doc)
    for group in hashes.values():
        if len(group) > 1:
            for doc in group:
                for page in doc.pages:
                    flag(doc, page, 'duplicate_document', page.canonical_start, page.canonical_end,
                         'Identical original bytes registered more than once.')
    for page in pages:
        page['issue_ids'] = [i.issue_id for i in issues if i.document_id == page['document_id'] and i.physical_page == page['physical_page']]
    return pages, sorted(issues, key=lambda i: (i.document_id, i.physical_page, i.start_char, i.code))


def propose_regions(documents, entries, issues):
    regions = []
    entry_map = {e.document_id: e for e in entries}
    for doc in documents:
        for page in doc.pages:
            text = doc.canonical_text[page.canonical_start:page.canonical_end]
            for a, b, part in blocks(text, page.canonical_start):
                # Split at diagnostic boundaries so a date conflict does not hold unrelated prose.
                relevant = [i for i in issues if i.document_id == doc.document_id and i.start_char < b and i.end_char > a]
                cuts = sorted({a, b, *(max(a, i.start_char) for i in relevant), *(min(b, i.end_char) for i in relevant)})
                for start, end in zip(cuts, cuts[1:]):
                    value = doc.canonical_text[start:end]
                    if not value.strip():
                        continue
                    span = EvidenceSpan(document_id=doc.document_id, content_sha256=doc.content_sha256,
                        canonical_sha256=doc.canonical_sha256, start_char=start, end_char=end)
                    intersects = [i.issue_id for i in relevant if i.start_char < end and i.end_char > start]
                    if len(re.findall(r'\w+', value)) < 8 or value.strip() in SECTION_NAMES:
                        fragment = Issue(issue_id='issue-' + fingerprint({'span': span.model_dump(), 'code': 'short_fragment'})[:20],
                            code='short_fragment', document_id=doc.document_id, physical_page=page.page_number,
                            start_char=start, end_char=end, detail='Short fragment or standalone heading; verify context or merge with contiguous reviewed evidence.')
                        if fragment.issue_id not in {i.issue_id for i in issues}:
                            issues.append(fragment)
                        if fragment.issue_id not in intersects:
                            intersects.append(fragment.issue_id)
                    regions.append(ReviewedRegion(region_id='region-' + fingerprint(span.model_dump())[:20],
                        evidence_span=span, physical_page=page.page_number,
                        printed_page_label=entry_map[doc.document_id].printed_page_labels.get(str(page.page_number)),
                        proposed_section=value.strip() if value.strip() in SECTION_NAMES else None,
                        recommendation='hold' if intersects else 'candidate', issue_ids=intersects))
    return regions
