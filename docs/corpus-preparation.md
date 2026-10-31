# Corpus preparation: current support and historical Phase 1.5 workflow

The core registry, audit, review and pilot loading modules below remain supported.
Subsequent work completed the frozen pilot, human sign-off, local gold export and
lexical/dense/RRF experiments. A conservative RAG module is now available; see
[the current README](../README.md) and [RAG contracts](rag.md). Statements below
about pending reviews or absent export tools describe the historical preparation
workflow, not the current dataset status. Local artifact provenance remains
authoritative; no public generic gold-export CLI is promised. Research artifacts
are still excluded from Git.

## Historical preparation workflow

This workflow prepares a **pilot**, not a complete research dataset. It does not
measure retrieval effectiveness, infer agricultural facts, approve evidence,
create ground-truth labels, run OCR, download sources, or train/run dense models.

## Commands and artifacts

Use the existing virtual environment and dependencies. No new packages are needed.
Run from the repository root:

```sh
source .venv/bin/activate
python -m pytest -q
python -m krishisaarthi.corpus --help
python -m krishisaarthi.corpus prepare --registry data/pilot/registry.jsonl --output data/pilot/review-v3
```

`prepare` requires a **fresh** output directory. Never rerun over an existing review
bundle. The supplied two-document registry and first audit are local under
`data/pilot/`; their contents are deliberately not in Git. On another checkout,
authorized users must supply their own originals and registry. The corpus CLI does not download documents. The separate dense model downloader is documented in dense-baselines.md.

Each bundle contains:

- `registry.jsonl`: frozen metadata snapshot, issue dates with month precision,
  publisher, known/unknown URLs, source languages, license status and inspection anchors.
- `originals/`, `extracted/`, `documents.jsonl`: byte-preserved original copies,
  canonical and per-page text, original/canonical hashes and extractor versions.
- `quality-report.json`, `quality-report.md`: document/page/region review queues.
  These reports do not reproduce source text, farmer names or address lists.
- `issues.jsonl`: immutable diagnostics with canonical spans and page references.
- `regions.jsonl`: editable proposals. Every initial status is `pending`, and every
  manual verification check is false. `candidate` is a recommendation, not approval.
- `integrity.json`: source/audit snapshot hashes and corpus fingerprint.
- `annotations/questions.csv`, `annotations/evidence.csv`: **header-only** templates;
  `annotations/INSTRUCTIONS.md` explains human annotation. No gold export is run.

The CLI permits research outputs only inside this repository's ignored `data/`
directory, rejects tracked destination paths, and refuses to overwrite bundles or
index builds. Both source PDFs and derived text, reports, annotations and index
artifacts stay local. Git ignore is not encryption or protection from `git add -f`.
Do not force-add research artifacts. Licenses are **unknown** until verified; public
access is not permission to redistribute originals, extracted text or filled forms.

## Document registry

The schema is `RegistryEntry` in `src/krishisaarthi/corpus/models.py`. Required fields
include document ID, relative/absolute original path, original SHA-256, title,
publication type, publisher, publication month, metadata evidence pages and source
languages. Record the publisher homepage separately from the exact download URL.
The latter remains null if unknown. Metadata embedded by a PDF authoring application
is not trusted as bibliographic information. PDF creation date is not issue date.

Printed labels are explicit observations keyed by physical page number, not assumed
from `PDF /PageLabels` or a constant offset. Unknown or absent printed labels are
null and must be checked manually. Physical page numbering is always one-based.

The original sources can be bilingual while the admitted knowledge base is English.
Their language list is preserved in the registry and `domain_metadata.source_languages`.
The reused Phase 1 `Document.language='en'` represents the intended index language;
it is **not** a certification that every source page is English.

`known_issue_anchors` records prior inspection findings, such as a conflicting date
or a repeated account. Each has a physical page, code, exact starting text, optional
`end_before`, and a text-free diagnostic explanation. An unmatched anchor flags the
whole page for re-review rather than silently dropping the issue. The March date
anchor holds the affected account, not all other evidence on its page. No source
character is corrected. Printed labels and anchors are observations, not approvals.

## Audit limitations

Diagnostics flag sparse/missing text, extraction exceptions, replacement characters,
known split-word patterns, script mixture, tables, likely personal identifiers,
non-wheat mentions, missing survey reports, running headers and duplicate material.
Images are counted; an image is not automatically classified as a scanned page.
Image-only information is not extracted. No automatic text repair is applied.

Near-duplicate comparison uses whitespace-normalized case-folded paragraph text,
minimum 180 characters, length ratio at least 0.65, and SequenceMatcher ratio at least
0.85. Normalization is used only for diagnostics, never canonical text or retrieval.
Original-file hashes identify exact duplicate documents. The paragraph comparison
is quadratic and intended for this small pilot, not a large-scale corpus service.

A candidate with no flags is **not known to be private-data-free, English, relevant,
correctly extracted or scientifically consistent**. Names without titles, addresses,
word splits, repeated facts with different wording and table structures can escape
heuristics. Human review is required for every admitted region. Table values must
remain held until a person verifies cell/row meaning from the original; this phase
does not reconstruct tables or create derived prose from them.

## Inspect and approve a region

1. Open the local quality report. Select a region ID and inspect its diagnostics:

   ```sh
   python -m krishisaarthi.corpus inspect --bundle data/pilot/review-v2 --region-id REGION_ID
   ```

2. Open the original PDF at the reported **physical page**. To read the exact local
   extracted slice, add `--show-text`. Its output may contain copyrighted source
   text or unreviewed personal information; do not publish it or use it as an example.
   The default inspector output contains offsets and diagnostics without source text.

3. Edit the matching record in `regions.jsonl`. Keep it `pending` while editing.
   Narrow/split regions to remove unnecessary personal names, private addresses,
   non-wheat content, unresolved table cells, duplicate text and inconsistent claims.
   For a split, give each record a unique region ID. Adjacent regions may be merged
   only when the entire contiguous source interval is reviewed; never join disjoint
   text by deleting intervening characters. All regions must stay inside one page.

4. Add `section_boundaries` as sorted, unique **absolute canonical offsets** strictly
   inside that region. Check them against the original section structure. Update the
   printed page label (null is valid for an absent label). Never edit canonical text,
   original files, frozen registry or issues to make the review pass.

5. Run `inspect` again on the final pending record. It reports the current
   `content_fingerprint_for_manual_review` and all intersecting audit issues, including
   issues missing from the editable record's `issue_ids`. For each remaining issue,
   enter a substantive human resolution in `resolutions`, keyed by issue ID. A note
   is not a substitute for resolving the underlying problem: leave the region
   pending or excluded if the source is still inconsistent, sensitive or ambiguous.

6. Only after manual verification, set all fields in `checks` to true:
   `english`, `wheat_relevant`, `privacy_clear`, `extraction_verified`,
   `duplicate_clear`, `consistency_clear`, `table_clear`, `printed_label_verified`.
   For a non-table region, `table_clear` confirms no unresolved table content.
   Enter your reviewer identifier, timezone-aware ISO timestamp, substantive
   `review_note`, and the inspector's fingerprint in `reviewed_content_fingerprint`.
   Finally set `status` to `approved`. Otherwise retain `pending` or use `excluded`.

Approval is a local human review record, not a digital signature or authentication
system. Fingerprints detect accidental changes; they do not defend against someone
intentionally rewriting the manifest. An edited span/boundary/printed label requires
new review and a new fingerprint. If an already approved record becomes invalid,
return it to `pending` before inspecting and reapproving it.

## Build only after your review

```sh
python -m krishisaarthi.corpus build --bundle data/pilot/review-v2 --output data/pilot/index-v1 --config configs/baseline.toml
```

The command fails without writing an index if no regions are manually approved.
It validates all source fingerprints, region bounds/page membership and the manual
checks; rejects overlapping approved regions, stale approvals, unresolved intersecting
issues and Devanagari in approved English regions; then chunks only approved spans.
No blanket language detector or personal-name detector replaces your judgment.

It reuses the Phase 1 chunker with **256-token size and 48-token overlap**. Windows
restart at region and explicit section boundaries and never span pages or exclusions.
Every passage is an exact canonical slice. Original page references are restored and
validated. Printed labels, region IDs and reviewer metadata travel in passage metadata.
The selection fingerprint and boundary policy are part of the derived chunking/ID
identity, so changes to reviewed selection create a distinct research index snapshot.

Outputs are `approved-passages.jsonl`, `approved-regions.jsonl` and
`index-manifest.json`. The frozen ledger provides the audit trail for that build.
Revoking approval requires a fresh build and retirement of the old local snapshot;
there is no server that automatically revokes already exported files.

Phase 1's unrestricted `chunk` and `search` commands remain available for backward
compatibility and fixtures. They do **not** enforce research approval. Use this
Phase 1.5 `build` output for the research index; do not use an unfiltered Phase 1
chunk file as approved research evidence. No retrieval model or scoring code changed.

## Manual multilingual annotation

Fill only human-verified English, Hindi and Hinglish questions. Equivalent translations
share a question_group_id and stay together in future dataset splits. Evidence uses
original/canonical hashes and source spans independently of chunk IDs, and should be
verified against the approved index and original PDF. Separate evidence rows support
multiple spans/pages. Preserve observation period and region context.

A `pending` record is not an unanswerable label. Unanswerability must be explicitly
verified against a frozen corpus. Neither blank templates nor candidate region files
are valid Phase 1 gold-label files. No research ground truth is automatically generated.

## Expanded pilot (Phase 1.6)

Registries also accept year-precision dates (`publication_date: "2023"`,
`date_precision: "year"`), geographic applicability and observed copyright
restrictions. Date precision must match the value; no month is inferred.

Optional `audit_page_numbers` is a sorted, unique, nonempty list of one-based
physical pages. The complete original and canonical extraction are retained;
only audit diagnostics and automatic region proposals are scoped to those pages.
Unselected pages are **not audited or approved**. Omission retains the previous
all-page behavior. Source page counts and audited page counts are reported
separately. This avoids treating a multi-crop publication as an all-wheat corpus.
Source-specific local shortlists can narrow the pending proposals without changing
canonical offsets or marking any checks as verified. A shortlist is not an index.

## Explicitly selected Pilot Corpus v0.1

The user-selected pilot has a separate admission record in local `selection.json`.
This records authorization to include exact source regions, not independent
agricultural correctness or current recommendation validity. Both remain
`not_established`. Previous pending review records, reviewer identities and manual
checks are not changed or fabricated. The existing named-reviewer build gate above
continues to work unchanged.

`corpus.pilot.build_selected_pilot` validates the exact allowlist, exclusions, source
hashes, canonical intervals, review fingerprints, page membership and unresolved
audit flags before reusing the same region chunker. Revised IDs cannot be replaced
by parent IDs. Applicability, conditions and previous source-matching references
travel with passages. Context references outside selected spans are metadata,
not indexed evidence or gold labels.

The local frozen `data/pilot-v0.1/` bundle preserves originals, canonical text,
selected regions, configuration, passages, fitted BM25 JSON statistics and a file
integrity ledger. Loading verifies hashes, reconstructs the existing BM25 model and
compares its fitted state exactly. This detects accidental changes, not deliberate
rewriting of both artifacts and their ledger. Keep the snapshot immutable; changes
require a new corpus version. Local preparation and validation records are in
`data/pilot-v0.1-preparation/` and are intentionally excluded from Git.

```sh
python -m krishisaarthi.corpus.pilot --bundle data/pilot-v0.1 --query 'wheat sowing depth' --k 3
```

This command emits result IDs, scores and offsets, not source passage text. Results
are unlabeled: zero-score ties must not be treated as relevant evidence. No scoring,
normalization, metrics or model-specific tokenization has changed.

`data/pilot-v0.1-annotations/` holds empty question/evidence templates and planning
instructions for 12–15 answerable and 3–5 explicitly unanswerable information needs.
English, Hindi and natural Hinglish variants share one group ID per need. Humans
must establish answerability and labels against the frozen spans. Source-region,
passage and gold-evidence IDs remain distinct. Blank rows are not unanswerability.

## Evaluation readiness and final sign-off

The report now uses schema version 2, adding per-language metrics and optional
retriever diagnostics. `configs/pilot-evaluation.toml` selects K = 1, 3, 5;
`configs/baseline.toml`, frozen corpus configuration and model settings are unchanged.
The common retrieval interface still requires only `index` and `search`. BM25 exposes
an optional `diagnostics(query, k)` hook; other retrievers report diagnostics as
unavailable instead of being assigned BM25 lexical semantics.

- `Recall@K` retains its previous span-fraction/passage-relevance behavior.
- `cumulative_evidence_mrr_at_k` still uses the first prefix fully covering ANY gold
  span. It does not measure whether all required evidence was retrieved.
- `complete_evidence_coverage_at_k` is 1 only if ALL required evidence spans are
  fully covered by the top-K interval union with matching document and version
  hashes. It is averaged over answerable span-labeled queries only. Adjacent and
  overlapping chunks may jointly cover a span; a character gap fails coverage.
- `mrr_at_k` remains conventional individually judged passage relevance. Complete
  evidence coverage is null for passage-only labels, not silently reinterpreted.

Reports contain separate `en`, `hi`, `hinglish` aggregates and an `unspecified`
category when needed, with query/exclusion counts. Empty denominators yield null.
Overall metrics retain query-macro aggregation. For the complete balanced pilot
this equals averaging languages within each group, then groups; missing variants
must be addressed at annotation validation, not silently treated as balanced.
Explicitly unanswerable queries are excluded from retrieval metrics but included in
score diagnostics. Missing/malformed labels still fail validation.

BM25 diagnostics count positive scores across the entire index, list no-positive
query IDs, and report their overall/per-language rates over all diagnostic queries.
They also list zero-score passage IDs and positive/zero/negative result counts among
the top max(K) results. Vocabulary overlap and empty-token flags distinguish a true
lexical no-match from matching terms whose IDF produces zero/negative scores.
A positive score is not a relevance judgment. Zero-score ties remain in the original
deterministic rankings and metric inputs for backward compatibility, but are never
counted as positive lexical matches. Thus a zero-score tie can accidentally cover
gold evidence; diagnostics must accompany interpretation of retrieval metrics.
No filtering, abstention, normalization or model changes are introduced.

Run software verification with `python -m pytest -q`. Tests use synthetic fixtures;
they do not run the labeled research pilot. The generic Phase 1 evaluator accepts
explicit GoldLabel records; it does not itself enforce this local human sign-off
form. No converter from pending annotations to gold has been created. Do not manually
bypass the pending-review gate by supplying generated proposals as verified labels.

### Record the existing review decisions

Use `data/annotation-final-signoff-v1/review-sheet.html` and edit only
`data/annotation-final-signoff-v1/reviewer-signoff.json`. Keep the candidate annotation
file unchanged: the form binds its SHA-256 and the frozen corpus fingerprint.

1. Review all three variants and candidate evidence for each of the 16 groups.
2. For each group set `decision` to `accept`, `revise`, or `reject`, and enter the
   genuinely reviewed `final_answerability`. The current proposals are answerable
   for IN-01–12 and unanswerable for IN-13–16; these are not prefilled final decisions.
3. Complete `english_accepted`, `hindi_accepted`, `hinglish_accepted`,
   `cross_language_equivalence_confirmed`, and `source_scope_and_conditions_accepted`
   with actual true/false judgments. Add unresolved concerns to `notes`.
4. For answerable groups complete `evidence_accepted` and
   `ALL_requirement_accepted`; IN-08 and IN-10 require BOTH spans. For unanswerable
   groups complete `corpus_relative_unanswerability_confirmed` after checking all
   frozen passages. Leave literal `not_applicable` fields unchanged.
5. Enter your actual reviewer identifier in `reviewer_identity` and actual review
   time with timezone in `reviewed_at_iso8601`. Do not infer or backdate either.
6. Only when all groups are accepted and all applicable checks are true, set
   top-level `final_decision` to `accept` and `status` to `signed_off`. Otherwise leave
   status pending and record needed revisions. A changed question or span requires
   a new annotation version and hash before signing; never change a hash just to
   make an unreviewed revision pass.

Recording sign-off does not automatically export gold or run evaluation. The next
explicit step is to validate the completed form against the exact candidate hash,
then prepare reviewed labels and run the requested baseline. Until then all research
annotations, raw documents and derived corpus artifacts remain local and Git-ignored.

### Extracted-page storage locations

Corpus validation binds a passage to its document ID, original-file hash, canonical
hash, physical page number and canonical interval. A local extracted-text path is
storage metadata, not page identity. Every declared extracted-page file must exist
and match the exact UTF-8 bytes of its canonical page slice, even when document and
passage paths are identical. Different paths are accepted only when both declared
files pass that check. A missing path on one side cannot replace a declared path on
the other. Original source-file bytes are also checked against the source hash.
In-memory ingestion with no declared page files on either side remains supported.
This check runs through the existing `validate_corpus` / `validate_labels` pathway;
it does not rewrite frozen records or weaken page/version/offset validation.
