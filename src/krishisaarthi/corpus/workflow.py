"""Local, fresh-directory corpus bundles with integrity-checked source snapshots."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
from ..config import fingerprint
from ..ingestion.storage import save_document, read_jsonl, write_jsonl
from ..schemas import Document
from ..evaluation.labels import unique_by
from .models import RegistryEntry, Issue, ReviewedRegion
from .audit import extract, audit, propose_regions, AUDIT_VERSION
from .annotations import write_templates


def private_path(path):
    """CLI research artifacts must live under ignored, untracked repository data/."""
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()).resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root / 'data'):
        raise ValueError('Research artifacts must stay under the ignored repository data/ directory')
    relative = str(resolved.relative_to(root))
    if subprocess.check_output(['git', 'ls-files', '--', relative], cwd=root).strip():
        raise ValueError('Research artifact path is tracked by Git')
    ignored = subprocess.run(['git', 'check-ignore', '-q', '--', relative], cwd=root)
    if ignored.returncode != 0:
        raise ValueError('Research artifact path is not protected by .gitignore')
    return resolved


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def prepare(registry_path, output):
    entries = read_jsonl(registry_path, RegistryEntry)
    if not entries:
        raise ValueError('Registry is empty')
    unique_by(entries, 'document_id')
    output.mkdir(parents=True, exist_ok=False)
    documents, diagnostics, failures = [], {}, []
    for entry in entries:
        source = (registry_path.parent / entry.path).resolve()
        try:
            doc, diagnostic = extract(entry, source)
            documents.append(save_document(doc, output))
            diagnostics[entry.document_id] = diagnostic
        except Exception as exc:
            failures.append({'document_id': entry.document_id, 'status': 'hold',
                             'reason': 'document_extraction_or_integrity_failure', 'error_type': type(exc).__name__})
    audit_documents, audit_diagnostics = [], {}
    entry_map = {e.document_id: e for e in entries}
    for doc in documents:
        selected = entry_map[doc.document_id].audit_page_numbers
        if selected is not None and not set(selected) <= {p.page_number for p in doc.pages}:
            raise ValueError('Audit page selection is outside the source PDF')
        pages = [p for p in doc.pages if selected is None or p.page_number in selected]
        audit_documents.append(doc.model_copy(update={'pages': pages}))
        audit_diagnostics[doc.document_id] = [d for d in diagnostics[doc.document_id]
            if selected is None or d['physical_page'] in selected]
    page_rows, issues = audit(audit_documents, entries, audit_diagnostics)
    regions = propose_regions(audit_documents, entries, issues)
    for page in page_rows:
        page['issue_ids'] = [i.issue_id for i in issues if i.document_id == page['document_id'] and i.physical_page == page['physical_page']]
    write_jsonl(output / 'documents.jsonl', documents)
    write_jsonl(output / 'registry.jsonl', entries)
    write_jsonl(output / 'issues.jsonl', issues)
    write_jsonl(output / 'regions.jsonl', regions)
    write_templates(output / 'annotations')
    per_document = []
    for entry in entries:
        selected = [r for r in regions if r.evidence_span.document_id == entry.document_id]
        own_pages = [p for p in page_rows if p['document_id'] == entry.document_id]
        per_document.append({'document_id': entry.document_id, 'title': entry.title,
            'license_status': entry.license_status, 'redistribution_permission': entry.redistribution_permission,
            'pages': len(own_pages),
            'source_pages': next((len(d.pages) for d in documents if d.document_id == entry.document_id), 0),
            'candidate_regions': sum(r.recommendation == 'candidate' for r in selected),
            'held_regions': sum(r.recommendation == 'hold' for r in selected), 'approved_regions': 0,
            'suitability': 'requires_manual_review' if own_pages else 'hold_extraction_failed'})
    report = {'schema_version': 1, 'audit_version': AUDIT_VERSION,
        'scope': 'Local pilot; audit restricted to registry-selected pages when specified. Not a research effectiveness evaluation.',
        'notice': 'Candidates are not approved. No source quotations or personal names are included in this report.',
        'documents': per_document, 'pages': page_rows,
        'issue_counts': dict(sorted(Counter(i.code for i in issues).items())),
        'failures': failures, 'approved_passages': 0}
    write_json(output / 'quality-report.json', report)
    lines = ['# Pilot corpus quality audit', '', report['notice'], '',
             'All page and region decisions are pending human review. Licenses remain as recorded in the registry.', '',
             '| Document | Audited physical pages | Candidate regions | Held regions | Approved |',
             '|---|---:|---:|---:|---:|']
    for row in per_document:
        lines.append(f"| {row['document_id']} | {row['pages']} | {row['candidate_regions']} | {row['held_regions']} | 0 |")
    lines += ['', '## Page review queue', '', '| Document | Physical page | Printed label (unverified) | Characters | Issues |', '|---|---:|---|---:|---|']
    for row in page_rows:
        codes = sorted({i.code for i in issues if i.issue_id in row['issue_ids']})
        lines.append(f"| {row['document_id']} | {row['physical_page']} | {row['printed_page_label'] or 'unknown/absent'} | {row['characters']} | {', '.join(codes) or 'Manual reading-order, privacy and scope review'} |")
    lines += ['', '## Region review queue', '', 'Offsets are canonical Unicode character offsets [start,end). No source text is reproduced.', '', '| Region | Document | Page | Start | End | Recommendation | Issues |', '|---|---|---:|---:|---:|---|---|']
    for region in regions:
        s = region.evidence_span
        codes = sorted({i.code for i in issues if i.issue_id in region.issue_ids})
        lines.append(f'| {region.region_id} | {s.document_id} | {region.physical_page} | {s.start_char} | {s.end_char} | {region.recommendation} (pending) | {", ".join(codes)} |')
    (output / 'quality-report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Snapshot everything except the intentionally editable human region review file.
    protected = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'regions.jsonl')
    integrity = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    write_json(output / 'integrity.json', {'schema_version': 1, 'sha256': integrity,
        'corpus_fingerprint': fingerprint(sorted((d.document_id, d.content_sha256, d.canonical_sha256) for d in documents))})
    return report


def load_bundle(bundle):
    integrity = json.loads((bundle / 'integrity.json').read_text(encoding='utf-8'))
    # Annotation templates may be filled locally; source/audit artifacts may not change.
    for name, expected in integrity['sha256'].items():
        if name.startswith('annotations/'):
            continue
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError('Invalid integrity path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Immutable source/audit artifact changed: {name}')
    for required in ('documents.jsonl', 'issues.jsonl', 'registry.jsonl'):
        if required not in integrity['sha256']:
            raise ValueError('Incomplete source/audit integrity manifest')
    return (read_jsonl(bundle / 'documents.jsonl', Document),
            read_jsonl(bundle / 'regions.jsonl', ReviewedRegion),
            read_jsonl(bundle / 'issues.jsonl', Issue), integrity)
