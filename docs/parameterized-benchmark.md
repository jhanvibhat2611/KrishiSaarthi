# Parameterized approved-corpus benchmark

`python -m krishisaarthi.evaluation.benchmark --config data/<run>/config.json --output data/<run>/report.json`

This is a separate runner, not a replacement for the frozen pilot scripts. It reads
snapshots produced by `python -m krishisaarthi.corpus build`. It cannot admit pending
regions or export annotations. Existing Pilot v0.1 uses its original explicit-selection
workflow and is not migrated by this runner. No Phase 6 real evaluation is authorized
until corpus and annotation sign-off is complete.

Configuration is `BenchmarkConfig` in `evaluation/benchmark.py`: corpus_version,
source_bundle, approved_snapshot, source_fingerprint, corpus_fingerprint,
approved_regions_sha256, index_manifest_sha256, annotations, annotations_sha256,
split (development/test), and settings. All paths resolve relative to the configuration.
Use source-bundle integrity's corpus_fingerprint and the approved index manifest's
passages_fingerprint as the two distinct corpus identifiers. Pin SHA-256 of the
approved-region ledger, index manifest, and signed annotation JSON. No placeholder
hash is valid. Configuration is persisted verbatim in the report. Defaults retain
256/48 chunks, K=1/3/5 and existing BM25 settings.

The loader validates immutable source artifacts and original/page contents, rebuilds
passages in memory from frozen approved-regions using the existing strict validator,
and compares complete passage records, manifest counts and fingerprints. It writes
no corpus/index changes. Zero approved regions fail. The CLI writes a fresh ignored
output only; never overwrite an old result. `run_benchmark(config, base, retriever)`
accepts the existing Retriever interface for already configured dense/hybrid adapters.
The initial CLI selects existing BM25, with no new ranking policy.

Annotation JSON follows `Dataset`: corpus_version/fingerprint; evidence catalog
with unique evidence_id and canonical span; groups with unique group_id, split,
evidence_family_ids, split_block_ids, explicit answerable, required_evidence_ids, ALL requirement
and signoff; queries with unique query_id, group_id, language, text, base/stress
variant_type and signoff. Signoff requires explicit accept, actual reviewer_id,
timezone-bearing reviewed_at and substantive note. This validates the record, not
the truth of an asserted human identity; it never manufactures signoff. Each group
requires exactly one base en/hi/hinglish query. Missing evidence or unreviewed rows
fail; blank is never interpreted as unanswerable. Evidence must fit the admitted
corpus. Only the loader constructs in-memory labels from already accepted records;
it never exports verified gold records or edits the input annotations.

Freeze evidence-family assignments before question assignment. Every shared family or connected split-block
ID must stay within one split; identical/overlapping canonical evidence cannot cross
splits even under different family IDs. Semantic cross-document duplicates cannot
be inferred from IDs/offsets: reviewers must assign the same family block, including
connected multi-span requirements. Current pilot families stay development-only.
A selected empty split fails; an explicitly unanswerable-only split is retained with
null positive-evidence metrics, not fabricated zero performance.

The historical schema-version-2 query-macro metrics remain unchanged. A new
`benchmark` section adds group-macro complete coverage at 1/3/5 and explicit primary
`group_macro_complete_evidence_coverage_at_3`: average across answerable groups of
the mean over their three base languages. ALL required spans must be fully covered
using the existing interval-union metric. With exactly three base variants per group,
base-only query macro equals group macro; stress variants never alter that primary
weighting. Stress rows are separately identified. The top-level historical metric
block remains query-macro over all supplied answerable variants. Base-language
reports and excluded group IDs are explicit. Recall and cumulative evidence MRR
retain existing definitions; cumulative MRR completes ANY evidence unit, not ALL.
No timing or uncertainty estimates are invented. Family-block bootstrap and new
retriever CLI selection are not implemented by this minimal runner.

Validation: `python -m pytest tests/test_benchmark.py -q` and full `pytest -q`.
Fixtures contain only synthetic source text and explicitly fictional signoff data.
