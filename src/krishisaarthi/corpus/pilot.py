"""Explicit user-selected pilot admission, distinct from claimed scientific verification."""
import argparse
import json
import re
from pathlib import Path
from ..config import fingerprint, load_config
from ..schemas import Passage, Query
from ..ingestion.storage import read_jsonl
from ..retrieval.bm25 import BM25Retriever
from .review import validate_regions, review_fingerprint, _chunk_regions
from .workflow import private_path


def build_selected_pilot(documents, regions, issues, config, selection):
    """Fail closed on substitutions; never manufacture human-review fields."""
    validate_regions(documents, regions, issues)
    if selection.get('authorization_basis') != 'explicit_user_selection':
        raise ValueError('Explicit user selection is required')
    if selection.get('independent_agricultural_correctness') != 'not_established' or selection.get('current_recommendation_validity') != 'not_established':
        raise ValueError('Pilot must preserve verification limitations')
    selected = selection['selected_regions']
    ids = [r['region_id'] for r in selected]
    if not ids or len(ids) != len(set(ids)) or len(ids) != selection['expected_region_count']:
        raise ValueError('Selection count or IDs invalid')
    if set(ids) & set(selection['excluded_region_ids']):
        raise ValueError('Excluded region selected')
    by_id = {r.region_id: r for r in regions}
    if set(by_id) != set(ids):
        raise ValueError('Input regions must exactly match selection allowlist')
    metadata = {}
    docs = {d.document_id: d for d in documents}
    for record in selected:
        region = by_id[record['region_id']]
        if record['review_fingerprint'] != review_fingerprint(region):
            raise ValueError('Selected fingerprint mismatch')
        if record['evidence_span'] != region.evidence_span.model_dump():
            raise ValueError('Selected source span mismatch')
        if region.status == 'excluded':
            raise ValueError('Excluded review record cannot enter pilot')
        s = region.evidence_span
        if any(i.document_id == s.document_id and i.start_char < s.end_char and i.end_char > s.start_char for i in issues):
            raise ValueError('Unresolved audit issue in selected region')
        text = docs[s.document_id].canonical_text[s.start_char:s.end_char]
        if not text.strip() or re.search(r'[\u0900-\u097f\x00-\x08\x0b\x0e-\x1f]', text):
            raise ValueError('Invalid English source text')
        if not record['source_matching_reference'] or not record['conditions_and_limitations']:
            raise ValueError('Missing source review context')
        metadata[region.region_id] = {
            'approval': 'explicit_user_selection_for_controlled_pilot',
            'pilot_version': selection['version'],
            'applicability': selection['applicability'],
            'source_matching_reference': record['source_matching_reference'],
            'conditions_and_limitations': record['conditions_and_limitations'],
            'context_references': record.get('context_references', []),
            'independent_agricultural_correctness': 'not_established',
            'current_recommendation_validity': 'not_established',
            'publication_allowed': False,
        }
    ordered = sorted(regions, key=lambda r: (r.evidence_span.document_id, r.evidence_span.start_char))
    for left, right in zip(ordered, ordered[1:]):
        if left.evidence_span.document_id == right.evidence_span.document_id and left.evidence_span.end_char > right.evidence_span.start_char:
            raise ValueError('Selected regions overlap')
    return _chunk_regions(documents, ordered, config, fingerprint(selection), metadata)


def bm25_state(retriever):
    """JSON-serializable fitted state, no unsafe pickle or hidden training."""
    m = retriever._model
    if m is None:
        raise ValueError('Index must be fitted')
    return {'retriever_id': retriever.retriever_id,
            'passage_ids': [p.passage_id for p in retriever._passages],
            'doc_freqs': m.doc_freqs, 'doc_len': m.doc_len, 'avgdl': m.avgdl,
            'idf': m.idf, 'average_idf': m.average_idf, 'corpus_size': m.corpus_size}


def load_pilot(bundle):
    """Verify frozen artifact hashes and deterministically rebuild the fitted BM25."""
    import hashlib
    ledger = json.loads((bundle / 'integrity.json').read_text())
    for name, expected in ledger.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen pilot artifact changed')
    for required in ['passages.jsonl', 'config.toml', 'selection.json', 'bm25-index.json', 'manifest.json']:
        if required not in ledger:
            raise ValueError('Incomplete frozen pilot')
    passages = read_jsonl(bundle / 'passages.jsonl', Passage)
    selection = json.loads((bundle / 'selection.json').read_text())
    allowed = {r['region_id']: r for r in selection['selected_regions']}
    for passage in passages:
        record = allowed.get(passage.domain_metadata.get('region_id'))
        if record is None:
            raise ValueError('Passage outside selected regions')
        s, p = record['evidence_span'], passage.evidence_span
        if (p.document_id, p.content_sha256, p.canonical_sha256) != (s['document_id'], s['content_sha256'], s['canonical_sha256']) or not s['start_char'] <= p.start_char < p.end_char <= s['end_char']:
            raise ValueError('Passage outside selected span')
    retriever = BM25Retriever(load_config(bundle / 'config.toml').bm25)
    retriever.index(passages)
    if bm25_state(retriever) != json.loads((bundle / 'bm25-index.json').read_text()):
        raise ValueError('Persisted BM25 state differs from rebuilt index')
    return passages, retriever


def main():
    parser = argparse.ArgumentParser(description='Search a frozen local pilot; no evaluation labels inferred')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    _, retriever = load_pilot(private_path(args.bundle))
    results = retriever.search(Query(query_id='unlabeled-local-query', text=args.query), args.k)
    print(json.dumps([r.model_dump() for r in results], indent=2))


if __name__ == '__main__':
    main()
