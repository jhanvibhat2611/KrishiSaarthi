"""Read-only QA-C3 review preparation; no model or retrieval execution."""
import copy
import json
import os
import secrets
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .qa_runtime import sha, proposal, validate_handoff, private_directory, private_write
from .qa_real import inputs, pack
from .qa_adapters import rag_prompt, validate_output, require
from .qa_comparison_fake import build_direct_prompt, fingerprint

FIELDS = {
    'factual_support': ['Fully supported by the authorized corpus', 'Partially supported', 'Unsupported by the authorized corpus', 'Uncertain / needs inspection', 'Not applicable — no answer generated'],
    'completeness': ['Complete', 'Partially complete', 'Incomplete', 'Not applicable'],
    'answerability': ['Appropriate answer', 'Appropriate withholding', 'Unsupported answer', 'Incorrect withholding', 'Uncertain', 'Not applicable — generation failure'],
    'language_quality': ['Appropriate', 'Partly appropriate', 'Inappropriate / unintelligible', 'Not applicable'],
}
CITATION = ['Citation actually supports the claim', 'Citation partially supports the claim', 'Citation does not support the claim', 'Uncertain']
EXPECTED = {'direct': (12, 8, 0, 8, 4), 'rag': (12, 10, 6, 4, 2)}


def snapshot(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def validate_records(results, handoff):
    """Reject missing/duplicate records and preserve failures without reparsing them as answers."""
    cases = {c['case_id']: c for c in handoff['cases']}
    rows = results['rows']
    expected = {(c, a) for c in cases for a in ('direct', 'rag')}
    require(len(cases) == 12 and len(rows) == 24 and {(r['case_id'], r['arm']) for r in rows} == expected, 'Pairing mismatch')
    require(results['model_attempts'] == 24 and results['reference_scoring_enabled'] is False, 'Run state mismatch')
    counts = {}
    for arm in ('direct', 'rag'):
        rs = [r for r in rows if r['arm'] == arm]
        counts[arm] = (sum(r['attempted'] is True for r in rs), sum(r['status'] != 'error' for r in rs),
                       sum(r['status'] == 'review_required' for r in rs), sum(r['status'] in ('uncertain', 'insufficient') for r in rs),
                       sum(r['status'] == 'error' for r in rs))
        require(counts[arm] == EXPECTED[arm], 'Outcome counts mismatch')
    for r in rows:
        require(r['language'] == cases[r['case_id']]['question']['language'] and isinstance(r['raw_output'], str) and bool(r['raw_output']), 'Missing output or language')
        require(r['status'] in ('review_required', 'uncertain', 'insufficient', 'error'), 'Unknown outcome')
        require(bool(r['failure_category']) == (r['status'] == 'error'), 'Failure category mismatch')
        if r['status'] == 'error':
            require(not r['claims'] and not r['citations'], 'Failure promoted to answer')
        if r['arm'] == 'direct':
            require(r['citation_applicability'] == 'not_applicable' and not r['citations'], 'Direct citation mismatch')
    return counts


def blank_review(row):
    return {'case_id': row['case_id'], 'arm': row['arm'], 'reviewed': False,
            **{k: None for k in FIELDS}, 'notes': '',
            'citation_applicability': 'not_applicable' if row['arm'] == 'direct' else 'provided_claims_only',
            'claim_citations': [{'claim_index': i, 'handle': h, 'support': None}
                for i, c in enumerate(row['claims']) for h in c.get('evidence_handles', [])] if row['arm'] == 'rag' else []}


def prepare(root, out):
    root = Path(root).resolve(); run = root/'data/qa-c3-real-development-v1'
    before = snapshot(run)
    read = lambda n: json.loads((run/n).read_text())
    results = read('results.json'); raw_h = read('handoff.json'); p = proposal(root)
    require(read('complete.json') == {'status': 'completed_once', 'reference_scoring_enabled': False, 'retry_permitted': False}, 'Incomplete run')
    require(sha(run/'handoff.json') == read('retrieval.json')['handoff_sha256'], 'Handoff hash mismatch')
    require(results['configuration'] == p.model_dump() and results['proposal_sha256'] == fingerprint(p.model_dump()), 'Proposal mismatch')
    for name, digest in read('execution-manifest.json')['code_sha256'].items():
        require(sha(root/name) == digest, 'Execution implementation fingerprint mismatch')
    marker = json.loads((root/'data/qa-c3-one-authorized-run.json').read_text())
    require(marker['directory'] == str(run) and marker['approval_sha256'] == read('authorization.json')['approval_sha256'], 'Authorization binding mismatch')
    docs, passages, questions = inputs(root, p)  # hashes only; never loads a model
    h = validate_handoff(raw_h, p, passages, questions, docs)
    counts = validate_records(results, raw_h)
    require(len(list((run/'attempts').glob('*.json'))) == 24, 'Attempt count mismatch')
    require(len(list((run/'retrieval-attempts').glob('*.json'))) == 12, 'Retrieval count mismatch')
    rows = {(r['case_id'], r['arm']): r for r in results['rows']}
    pairs = []
    for case in h.cases:
        evidence = pack(case, h, p)
        e5 = read('retrieval-attempts/'+case.case_id+'--e5.json')
        require(e5['record']['passage_ids'] == [x.passage_id for x in case.passages], 'Retrieved ordering mismatch')
        for arm in ('direct', 'rag'):
            row = rows[(case.case_id, arm)]
            journal = read('attempts/'+case.case_id+'--'+arm+'.json')
            require(journal['state'] == 'completed_record' and journal['record'] == row, 'Journal mismatch')
            prompt = build_direct_prompt(case.question) if arm == 'direct' else rag_prompt(case.question, evidence)
            require(row['prompt'] == prompt and row['prompt_sha256'] == fingerprint(prompt), 'Prompt mismatch')
            require(row['evidence_fingerprint'] == (None if arm == 'direct' else evidence.fingerprint), 'Evidence binding mismatch')
            if row['status'] != 'error':
                parsed = validate_output(row['raw_output'], arm, evidence)
                require(all(row[k] == parsed[k] for k in ('status', 'claims', 'citations', 'reason')), 'Parsed outcome mismatch')
        sources = json.loads(rag_prompt(case.question, evidence)[1]['content'])['sources']
        pairs.append({'case_id': case.case_id, 'question': case.question.model_dump(),
                      'rows': [rows[(case.case_id, a)] for a in ('direct', 'rag')],
                      'evidence': [{**s, 'provenance': x.model_dump(exclude={'text'})} for s, x in zip(sources, case.passages)]})
    # Entire frozen collection is available for corpus-relative withholding review, not only Top-5.
    corpus = [x.model_dump() for x in passages]
    binding = {'run_id': run.name, 'corpus_fingerprint': h.corpus_fingerprint,
               'proposal_sha256': results['proposal_sha256'], 'artifact_sha256': before,
               'frozen_documents_sha256': sha(root/'data/pilot-v0.1/documents.jsonl'),
               'frozen_passages_sha256': sha(root/'data/pilot-v0.1/passages.jsonl')}
    package = {'binding': binding, 'counts': counts, 'pairs': pairs, 'corpus': corpus,
               'document_metadata': {d.document_id: d.domain_metadata for d in docs},
               'rubric': FIELDS, 'citation_rubric': CITATION,
               'reviews': [blank_review(r) for r in results['rows']]}
    package['package_fingerprint'] = fingerprint(package)
    require(snapshot(run) == before, 'Original run changed')
    out = private_directory(root, out)
    private_write(out/'package.json', package)
    template = Path(__file__).with_name('qa_review.html').read_text()
    # JSON embedded as inert data; escape HTML close tags and Unicode JS separators.
    payload = json.dumps(package, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    html = template.replace('__PACKAGE__', payload)
    fd = os.open(out/'review.html', os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f: f.write(html)
    private_write(out/'integrity.json', {'original_run_unchanged': True, 'binding': binding,
                  'package_sha256': sha(out/'package.json'), 'html_sha256': sha(out/'review.html'),
                  'reference_drafts_loaded': False, 'automatic_scores': False})
    return package


def validate_decisions(payload, package):
    require(set(payload) == {'package_fingerprint', 'reviewer_id', 'records'}, 'Unexpected export fields')
    require(payload['package_fingerprint'] == package['package_fingerprint'], 'Wrong review package')
    require(isinstance(payload['reviewer_id'], str) and 0 < len(payload['reviewer_id'].strip()) <= 200, 'Enter a reviewer identifier')
    records = payload['records']; expected = {(r['case_id'], r['arm']): r for r in package['reviews']}
    require(isinstance(records, list) and len(records) == 24, '24 review records required')
    require({(r['case_id'], r['arm']) for r in records} == set(expected), 'Duplicate or missing review record')
    for r in records:
        base = expected[(r['case_id'], r['arm'])]
        require(set(r) == set(base) and type(r['reviewed']) is bool, 'Invalid review fields')
        require(r['citation_applicability'] == base['citation_applicability'], 'Citation applicability changed')
        require(isinstance(r['notes'], str) and len(r['notes']) <= 10000, 'Invalid notes')
        for field, choices in FIELDS.items():
            require(r[field] is None or r[field] in choices, 'Invalid score')
            if r['reviewed']: require(r[field] is not None, 'Reviewed record has blank scores')
        cc = r['claim_citations']; bc = base['claim_citations']
        require(len(cc) == len(bc), 'Claim citation count changed')
        for c, b in zip(cc, bc):
            require(set(c) == set(b) and c['claim_index'] == b['claim_index'] and c['handle'] == b['handle'], 'Claim citation binding changed')
            require(c['support'] is None or c['support'] in CITATION, 'Invalid claim score')
            if r['reviewed']: require(c['support'] is not None, 'Reviewed citation has blank support')
    return copy.deepcopy(payload)


def save_export(directory, payload, package):
    value = validate_decisions(payload, package)
    value.update(schema_version=1, binding=package['binding'],
                 exported_at=datetime.now(timezone.utc).isoformat(),
                 timestamp_meaning='Time of explicit export; not inferred per-response review time',
                 reviewed_count=sum(r['reviewed'] for r in value['records']),
                 reference_approval=False, automatic_publication=False, externally_verified_false=None)
    dest = directory/'exports'; dest.mkdir(mode=0o700, exist_ok=True)
    name = 'human-review-'+uuid.uuid4().hex+'.json'
    private_write(dest/name, value)
    return name


def serve(directory, port=8774):
    directory = Path(directory).resolve()
    package = json.loads((directory/'package.json').read_text())
    integrity = json.loads((directory/'integrity.json').read_text())
    require(sha(directory/'package.json') == integrity['package_sha256'] and sha(directory/'review.html') == integrity['html_sha256'], 'Review package changed')
    root = Path(__file__).resolve().parents[3]
    require(directory.is_relative_to(root/'data') and directory.stat().st_mode & 0o777 == 0o700, 'Private data directory required')
    run = root/'data'/package['binding']['run_id']
    require(run == root/'data/qa-c3-real-development-v1' and snapshot(run) == package['binding']['artifact_sha256'], 'Original run changed since preparation')
    for name in ('documents', 'passages'):
        require(sha(root/('data/pilot-v0.1/'+name+'.jsonl')) == package['binding']['frozen_'+name+'_sha256'], 'Frozen corpus changed')
    token = secrets.token_hex(32)
    html = (directory/'review.html').read_text().replace('__SESSION_TOKEN__', token).encode()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def send(self, status, data, mime):
            self.send_response(status); self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.end_headers(); self.wfile.write(data)
        def allowed(self): return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'
        def do_GET(self):
            if self.allowed() and self.path in ('/', '/review.html'): self.send(200, html, 'text/html; charset=utf-8')
            else: self.send(404, b'Not found', 'text/plain')
        def do_POST(self):
            if not self.allowed() or self.path != '/export' or self.headers.get('Origin') != f'http://127.0.0.1:{self.server.server_port}' or self.headers.get('X-Review-Token') != token:
                self.send(403, b'Local session required', 'text/plain'); return
            try:
                n = int(self.headers.get('Content-Length', '0'))
                require(0 < n <= 500000, 'Invalid export size')
                name = save_export(directory, json.loads(self.rfile.read(n)), package)
                self.send(200, json.dumps({'saved': 'exports/'+name}).encode(), 'application/json')
            except (ValueError, KeyError, TypeError, OSError): self.send(400, b'Export rejected; check reviewer identifier and review fields.', 'text/plain')
    server = HTTPServer(('127.0.0.1', port), Handler)
    print(f'Private human review: http://127.0.0.1:{server.server_port}/review.html', flush=True)
    server.serve_forever()
