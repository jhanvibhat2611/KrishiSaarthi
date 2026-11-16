"""Versioned approved-corpus evaluation. No annotation export or source admission.

The existing query-macro runner/metrics are unchanged. This module adds explicit
base-language group weighting and validates declared evidence-family split blocks.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator

from ..schemas import Record, Query, GoldLabel, EvidenceSpan, Passage
from ..config import fingerprint, Config, EvaluationConfig
from ..corpus.workflow import load_bundle, private_path
from ..corpus.models import ReviewedRegion
from ..corpus.review import build_passages
from ..ingestion.storage import read_jsonl
from ..retrieval.bm25 import BM25Retriever
from .labels import unique_by, validate_labels
from .runner import evaluate, aggregate_metrics


class Signoff(Record):
    decision: Literal['accept']
    reviewer_id: str = Field(min_length=1)
    reviewed_at: str
    note: str = Field(min_length=1)

    @model_validator(mode='after')
    def genuine_fields(self):
        if not self.reviewer_id.strip() or not self.note.strip():
            raise ValueError('Blank sign-off fields')
        if datetime.fromisoformat(self.reviewed_at).tzinfo is None:
            raise ValueError('Sign-off timestamp requires timezone')
        return self


class BenchmarkQuery(Query):
    group_id: str = Field(min_length=1)
    variant_type: Literal['base', 'stress'] = 'base'
    signoff: Signoff


class EvidenceRecord(Record):
    evidence_id: str = Field(min_length=1)
    span: EvidenceSpan


class NeedGroup(Record):
    group_id: str = Field(min_length=1)
    split: Literal['development', 'test']
    evidence_family_ids: list[str] = Field(min_length=1)
    split_block_ids: list[str] = Field(min_length=1)
    answerable: bool
    required_evidence_ids: list[str]
    evidence_requirement: Literal['ALL'] = 'ALL'
    signoff: Signoff

    @model_validator(mode='after')
    def identifiers(self):
        for values in (self.evidence_family_ids, self.split_block_ids, self.required_evidence_ids):
            if any(not s.strip() for s in values) or len(values) != len(set(values)):
                raise ValueError('Missing or duplicate annotation identifiers')
        if self.answerable != bool(self.required_evidence_ids):
            raise ValueError('Answerability requires explicit matching evidence')
        return self


class Dataset(Record):
    corpus_version: str = Field(min_length=1)
    corpus_fingerprint: str
    groups: list[NeedGroup] = Field(min_length=1)
    queries: list[BenchmarkQuery] = Field(min_length=1)
    evidence: list[EvidenceRecord]


class BenchmarkConfig(Record):
    corpus_version: str = Field(min_length=1)
    source_bundle: str
    approved_snapshot: str
    source_fingerprint: str
    corpus_fingerprint: str
    approved_regions_sha256: str
    index_manifest_sha256: str
    annotations: str
    annotations_sha256: str
    split: Literal['development', 'test'] = 'development'
    settings: Config = Field(default_factory=lambda: Config(evaluation=EvaluationConfig(ks=[1, 3, 5])))

    @model_validator(mode='after')
    def fixed_protocol(self):
        if self.settings.evaluation.ks != [1, 3, 5]:
            raise ValueError('Benchmark protocol requires K=1,3,5')
        if (self.settings.chunking.size, self.settings.chunking.overlap) != (256, 48):
            raise ValueError('Benchmark protocol requires 256/48 chunking')
        for name in ('source_fingerprint', 'corpus_fingerprint', 'approved_regions_sha256',
                     'index_manifest_sha256', 'annotations_sha256'):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Expected SHA256 fingerprint')
        return self


def checked_file(path, expected):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'Immutable file fingerprint mismatch: {path.name}')
    return payload


def load_snapshot(config, base):
    """Rebuild in memory through approved-region validation, then compare exact records.

    Source and snapshot directories are inputs only. No index/artifacts are written.
    Frozen approved-regions, not mutable source-bundle regions, govern admission.
    """
    documents, _, issues, ledger = load_bundle((base / config.source_bundle).resolve())
    if (ledger['corpus_fingerprint'] != config.source_fingerprint or
            fingerprint(sorted((d.document_id, d.content_sha256, d.canonical_sha256) for d in documents)) != config.source_fingerprint):
        raise ValueError('Source corpus fingerprint mismatch')
    snap = (base / config.approved_snapshot).resolve()
    checked_file(snap / 'approved-regions.jsonl', config.approved_regions_sha256)
    manifest = json.loads(checked_file(snap / 'index-manifest.json', config.index_manifest_sha256))
    regions = read_jsonl(snap / 'approved-regions.jsonl', ReviewedRegion)
    if not regions or any(r.status != 'approved' for r in regions):
        raise ValueError('Snapshot requires nonempty approved regions')
    passages = read_jsonl(snap / 'approved-passages.jsonl', Passage)
    rebuilt, selection = build_passages(documents, regions, issues, config.settings.chunking)
    actual = fingerprint([p.model_dump() for p in passages])
    if actual != config.corpus_fingerprint or manifest['passages_fingerprint'] != actual:
        raise ValueError('Corpus fingerprint mismatch')
    if [p.model_dump() for p in rebuilt] != [p.model_dump() for p in passages]:
        raise ValueError('Snapshot differs from approved provenance/chunking')
    if (manifest['selection_fingerprint'] != selection or
            manifest['source_corpus_fingerprint'] != config.source_fingerprint or
            manifest['chunking'] != config.settings.chunking.model_dump() or
            manifest['passages'] != len(passages) or manifest['approved_regions'] != len(regions)):
        raise ValueError('Inconsistent snapshot manifest')
    return documents, passages


def validate_dataset(dataset, config, documents, passages):
    if (dataset.corpus_version != config.corpus_version or
            dataset.corpus_fingerprint != config.corpus_fingerprint):
        raise ValueError('Annotation corpus version/fingerprint mismatch')
    for records, field in ((dataset.groups, 'group_id'), (dataset.queries, 'query_id'), (dataset.evidence, 'evidence_id')):
        if any(not getattr(r, field).strip() for r in records):
            raise ValueError('Blank annotation identifier')
    groups = unique_by(dataset.groups, 'group_id')
    unique_by(dataset.queries, 'query_id')
    evidence = unique_by(dataset.evidence, 'evidence_id')
    families, used, group_queries = {}, set(), {g: [] for g in groups}
    labels = []
    for group in groups.values():
        for kind, identifiers in (('family', group.evidence_family_ids), ('block', group.split_block_ids)):
            for identifier in identifiers:
                key = (kind, identifier)
                if key in families and families[key] != group.split:
                    raise ValueError('Evidence-family or split-block leakage across development/test')
                families[key] = group.split
        if not set(group.required_evidence_ids) <= evidence.keys():
            raise ValueError('Unknown evidence ID')
        used.update(group.required_evidence_ids)
        spans = [evidence[e].span for e in group.required_evidence_ids]
        if len({s.model_dump_json() for s in spans}) != len(spans):
            raise ValueError('Duplicate evidence intervals under different IDs')
    if used != evidence.keys():
        raise ValueError('Unused or inconsistent evidence identifiers')
    for query in dataset.queries:
        if query.group_id not in groups or query.language not in ('en', 'hi', 'hinglish'):
            raise ValueError('Unknown group or missing language')
        group = groups[query.group_id]
        group_queries[query.group_id].append(query)
        labels.append(GoldLabel(query_id=query.query_id, answerable=group.answerable,
            evidence_spans=[evidence[e].span for e in group.required_evidence_ids]))
    for qs in group_queries.values():
        langs = [q.language for q in qs if q.variant_type == 'base']
        if sorted(langs) != ['en', 'hi', 'hinglish']:
            raise ValueError('Exactly one base variant per language is required per group')
    # Catch falsely separated IDs for overlapping actual evidence as well.
    for i, left in enumerate(dataset.groups):
        for right in dataset.groups[i+1:]:
            if left.split == right.split:
                continue
            for a in left.required_evidence_ids:
                for b in right.required_evidence_ids:
                    x, y = evidence[a].span, evidence[b].span
                    if ((x.document_id, x.content_sha256, x.canonical_sha256) ==
                        (y.document_id, y.content_sha256, y.canonical_sha256) and
                        x.start_char < y.end_char and y.start_char < x.end_char):
                        raise ValueError('Overlapping evidence leaks across split families')
    _, _, warnings = validate_labels(dataset.queries, labels, documents, passages)
    if warnings:
        raise ValueError('Required evidence not coverable by approved snapshot')
    if not any(g.split == config.split for g in dataset.groups):
        raise ValueError('Selected split is empty')
    return labels


def group_macro(rows, queries, groups, ks=(1, 3, 5)):
    """Base EN/HI/Hinglish only; stress rows never change primary denominator."""
    by_query = {q.query_id: q for q in queries}
    if len(by_query) != len(queries):
        raise ValueError('Duplicate query identifiers')
    grouped = {}
    seen = set()
    for row in rows:
        qid = row['query_id']
        if qid not in by_query or qid in seen:
            raise ValueError('Missing/duplicate result query identifier')
        seen.add(qid)
        q = by_query[qid]
        if q.variant_type == 'base':
            grouped.setdefault(q.group_id, []).append(row)
    expected = {g.group_id for g in groups if g.answerable}
    if set(grouped) != expected:
        raise ValueError('Missing answerable group results')
    metrics, per_group = {}, {}
    for gid in sorted(grouped):
        subset = grouped[gid]
        if sorted(r['language'] for r in subset) != ['en', 'hi', 'hinglish']:
            raise ValueError('Incomplete base languages in group results')
        per_group[gid] = aggregate_metrics(subset, ks)
    for k in ks:
        metrics[str(k)] = (sum(m['complete_evidence_coverage_at_k'][str(k)]
                                  for m in per_group.values()) / len(per_group)) if per_group else None
    return {'complete_evidence_coverage_at_k': metrics,
            'group_macro_complete_evidence_coverage_at_3': metrics['3'],
            'answerable_groups': len(per_group), 'per_group': per_group,
            'description': 'Mean over answerable groups of mean over three base languages; stress variants excluded.'}


def run_benchmark(config, base, retriever):
    documents, passages = load_snapshot(config, base)
    dataset = Dataset.model_validate_json(checked_file((base / config.annotations).resolve(), config.annotations_sha256))
    labels = validate_dataset(dataset, config, documents, passages)
    groups = [g for g in dataset.groups if g.split == config.split]
    group_ids = {g.group_id for g in groups}
    queries = sorted((q for q in dataset.queries if q.group_id in group_ids), key=lambda q: q.query_id)
    ids = {q.query_id for q in queries}
    labels = sorted((l for l in labels if l.query_id in ids), key=lambda l: l.query_id)
    report = evaluate(retriever, queries, labels, documents, passages, config.settings.evaluation)
    base_ids = {q.query_id for q in queries if q.variant_type == 'base'}
    base_rows = [r for r in report['per_query'] if r['query_id'] in base_ids]
    report['benchmark'] = {'schema_version': 1, 'configuration': config.model_dump(),
        'corpus_version': config.corpus_version, 'corpus_fingerprint': config.corpus_fingerprint,
        'annotations_sha256': config.annotations_sha256, 'split': config.split,
        'group_macro': group_macro(report['per_query'], queries, groups),
        'base_per_language': {lang: aggregate_metrics([r for r in base_rows if r['language'] == lang], [1,3,5])
                              for lang in ('en','hi','hinglish')},
        'stress_per_query': [r for r in report['per_query'] if r['query_id'] not in base_ids],
        'excluded_groups': [g.group_id for g in groups if not g.answerable],
        'split_validation_limit': 'Declared family IDs and overlapping spans checked; semantic near-duplicate family assignment requires human review.'}
    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate an approved snapshot and signed dataset; never exports labels')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = BenchmarkConfig.model_validate_json(args.config.read_text())
    output = private_path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite prior experiment')
    report = run_benchmark(config, args.config.resolve().parent, BM25Retriever(config.settings.bm25))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
