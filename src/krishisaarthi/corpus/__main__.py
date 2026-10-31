"""Usage: python -m krishisaarthi.corpus --help"""
import argparse
import json
from pathlib import Path
from ..config import load_config, fingerprint
from ..ingestion.storage import write_jsonl
from .workflow import prepare, private_path, load_bundle, write_json
from .review import build_passages, review_fingerprint, validate_regions


def main():
    parser = argparse.ArgumentParser(description='Local pilot corpus preparation; no automatic approvals')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--registry', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--bundle', type=Path, required=True)
    inspect.add_argument('--region-id', required=True)
    inspect.add_argument('--show-text', action='store_true', help='Local copyrighted/unreviewed text: do not publish or paste as examples')
    build = sub.add_parser('build')
    build.add_argument('--bundle', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--config', type=Path, default=Path('configs/baseline.toml'))
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            private_path(args.registry)
            report = prepare(args.registry.resolve(), private_path(args.output))
            print(json.dumps({'documents': report['documents'], 'failures': report['failures'],
                              'approved_passages': 0}, indent=2))
            return
        bundle = private_path(args.bundle)
        documents, regions, issues, integrity = load_bundle(bundle)
        if args.command == 'inspect':
            # Pending records are validated too; pending is never an index approval.
            validate_regions(documents, regions, issues)
            region = next((r for r in regions if r.region_id == args.region_id), None)
            if region is None:
                raise ValueError('Unknown region ID')
            doc = next(d for d in documents if d.document_id == region.evidence_span.document_id)
            span = region.evidence_span
            relevant = [i for i in issues if i.document_id == doc.document_id and
                        i.start_char < span.end_char and i.end_char > span.start_char]
            result = {'region': region.model_dump(), 'content_fingerprint_for_manual_review': review_fingerprint(region),
                      'intersecting_issues': [i.model_dump() for i in relevant],
                      'original_pdf': doc.original_file_path, 'notice': 'Inspection does not approve this region.'}
            if args.show_text:
                result['unreviewed_local_source_text'] = doc.canonical_text[span.start_char:span.end_char]
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            output = private_path(args.output)
            config = load_config(args.config)
            passages, selection_hash = build_passages(documents, regions, issues, config.chunking)
            output.mkdir(parents=True, exist_ok=False)
            write_jsonl(output / 'approved-passages.jsonl', passages)
            write_json(output / 'index-manifest.json', {'schema_version': 1,
                'scope': 'pilot_only', 'source_corpus_fingerprint': integrity['corpus_fingerprint'],
                'selection_fingerprint': selection_hash, 'chunking': config.chunking.model_dump(),
                'approved_regions': sum(r.status == 'approved' for r in regions), 'passages': len(passages),
                'passages_fingerprint': fingerprint([p.model_dump() for p in passages]),
                'redistribution_permission': 'not_established', 'labels_generated': False})
            # Freeze the approved review ledger alongside this specific index build.
            write_jsonl(output / 'approved-regions.jsonl', [r for r in regions if r.status == 'approved'])
            print(f'Built {len(passages)} passages from manually approved regions. No evaluation labels created.')
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
