# KrishiSaarthi — Phase 1 retrieval baseline

A modular Python scaffold for traceable ingestion, deterministic passage chunking,
BM25 retrieval and gold-evidence evaluation. Initial scope: **wheat cultivation and
crop management in India**, **English source documents**, and **English, Hindi and
Hindi-English (Hinglish) queries**.

No dataset has been downloaded. No dense model, query normalization, training,
RAG generation, verification pipeline or UI is implemented. Empty packages reserve
those component boundaries for later approved work.

## Setup

Run commands from the repository root. Python 3.11+ is required. The tested host
uses Python 3.12.10 on macOS arm64. The dependency lock pins runtime, test and build
dependencies; it does not contain machine-specific editable paths.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -m pytest -q
```

The lock is the reproducible setup. `pyproject.toml` also declares supported
dependency ranges. Regenerate a lock only intentionally and rerun the tests.

## Synthetic demonstration

**Everything in `tests/fixtures` is synthetic development material. Its scores
are software demonstrations, not research findings or evidence of retrieval quality.**
The placeholder `example.invalid` URL is not a research citation. The Hindi query
has no lexical match to the English evidence; its position depends on tie-breaking,
not cross-lingual understanding.

```sh
krishisaarthi ingest --manifest tests/fixtures/manifest.jsonl --output-dir data/demo
krishisaarthi chunk --documents data/demo/documents.jsonl --output data/demo/passages/passages.jsonl
krishisaarthi search --passages data/demo/passages/passages.jsonl --query seedmarker --k 3
krishisaarthi evaluate --documents data/demo/documents.jsonl --passages data/demo/passages/passages.jsonl --queries tests/fixtures/synthetic_queries.jsonl --gold tests/fixtures/synthetic_gold.jsonl --output data/demo/runs/bm25-fixture.json
```

The fixture workflow contains two documents, eight passages, three answerable
queries and one explicitly unanswerable query. The JSON report includes per-query
rankings, aggregate metrics, exclusion reasons, coverage semantics and a run manifest.
It is safe to rerun in the same output directory; named report files are replaced.

Regenerate fixture source files and deliberately authored labels with:

```sh
python scripts/make_fixtures.py
```

Gold fixture spans come from exact, authored sentences, independently of chunk IDs
and retrieval output. Regeneration is a development operation, not an annotation
procedure for research data.

## Structure and component interfaces

```text
configs/baseline.toml          Chunking, BM25 and evaluation settings
src/krishisaarthi/
  schemas.py                  Versioned Pydantic records
  config.py                   Validated TOML and fingerprints
  ingestion/                  PDF/text adapters, preservation, JSONL storage
  preprocessing/              Unicode token spans and deterministic chunking
  retrieval/                  Retriever protocol and BM25 implementation
  evaluation/                 Label/corpus validation, metrics, runner
  query_transformation/       Reserved; not implemented
  generation/                 Reserved; not implemented
  verification/               Reserved; not implemented
  cli.py                      Separate ingest/chunk/search/evaluate commands
scripts/make_fixtures.py       Reproducible synthetic fixture authoring
tests/                        Unit tests and subprocess CLI integration test
```

- `DocumentIngestor.ingest(path, metadata) -> Document` extracts without chunking.
- `SpanTokenizer.token_spans(text) -> Sequence[TokenSpan]` returns original offsets.
- `PassageChunker.chunk(document) -> Sequence[Passage]` slices canonical text.
- `Retriever.index(passages)` builds an index; `search(query, k)` returns ranked
  `RetrievalResult` records. Ranks are consecutive and one-based; IDs are unique.
- `evaluate(retriever, queries, gold, documents, passages, config)` validates,
  indexes and evaluates any implementation of the retrieval protocol. A fixed-order
  test retriever demonstrates independence from BM25.

Future Multilingual E5 and BGE-M3 adapters must use their own model tokenizers,
check all input token limits and **never silently truncate passages**. Their query
prefixes, pooling and scoring behavior require explicit configurations. None of
these model behaviors are implemented in this phase.

## Ingestion and provenance

An input manifest is JSONL. Each row contains a local `path` and optional `metadata`.
Relative paths resolve against the manifest directory. For example:

```json
{"path":"approved-wheat.pdf","metadata":{"document_id":"wheat-source-001","source_url":"https://publisher.example/document","language":"en","domain_metadata":{"crop":"wheat","country":"India","region":"provided region","topic":"provided topic"}}}
```

The URL above illustrates schema syntax; it is not a real source or citation.
No URL is fetched. Metadata is supplied by the caller, not inferred from documents.
English is the declared source-language constraint, not automatic language detection.
Crop/country defaults apply when domain metadata is omitted; supplied metadata is
retained as supplied. Duplicate document IDs in one corpus are rejected.

`Document` retains document ID, original filename/path, source URL, media type,
language, domain metadata, original SHA-256, canonical-text SHA-256, canonical text,
page references and extraction metadata. Default document IDs hash original bytes;
caller-provided stable IDs are supported. File copies and page extraction artifacts
are written under the output directory using safe hashed document directory names.
The document JSONL is the authoritative metadata manifest.

Original files are copied byte-for-byte. UTF-8 text decoding preserves CRLF and BOMs.
PDF pages use pypdf's plain extraction order without cleanup. Canonical PDF text
joins page strings with `\n\f\n`; this policy and extractor version are recorded.
Blank pages remain in page references and are listed in extraction metadata.
PDFs with no extractable text and encrypted PDFs are rejected. OCR is out of scope.

All offsets are **zero-based, half-open Python Unicode character offsets**, not
byte offsets, UTF-16 offsets or PDF coordinates. `canonical_text[start:end]`
reconstructs passage text exactly. PDF page numbers are one-based; text files use
null page numbers. PDF page references are not bounding boxes or visual layout spans.

## Chunking and BM25

The default is 256 baseline tokens with 48-token overlap. These tokens are Unicode
letter/mark/number runs plus individual non-whitespace punctuation characters.
Devanagari combining marks are retained. Chunking and retrieval share this baseline
token definition; BM25 additionally case-folds terms. No stemming, stopword removal,
transliteration, translation or query normalization is performed.

Chunks are page-bounded and split at explicit Markdown-style heading lines by
default. The heading regex is configurable; PDF font size or visual appearance is
not interpreted as a heading. Section titles remain part of the following section.
Overlap restarts at page/section boundaries. Cross-page merging is disabled because
no general deterministic continuity rule has been approved. A future such rule
must be versioned and explicitly approved.

Whitespace at token boundaries is retained to avoid artificial holes in strict
coverage, including when overlap is zero. Whitespace-only sections produce no
passages. Synthetic separators between pages are not indexed. A gold span crossing
such a separator may therefore be impossible to cover; evaluation reports this
warning and retains the span in the denominator. Annotate page-local gold spans
when evidence occupies separate pages.

Passage IDs hash document identity, original/canonical fingerprints, canonical
span and chunking configuration. Passage records also retain page references,
filename, source URL and domain metadata. Configuration IDs include tokenizer and
boundary-policy versions. Changed chunking creates new IDs; canonical gold spans
remain independent of those IDs.

BM25 uses `rank-bm25`'s Okapi implementation with configurable `k1`, `b` and `epsilon`.
The index is rebuilt in memory from saved passages. Equal scores are ordered by
passage ID. All-zero queries still return deterministic ranked candidates. Scores
are not probabilities, and the library's IDF handling can produce negative scores
in very small or repetitive corpora.

To change experiment settings, copy the config and pass it before the subcommand:

```sh
krishisaarthi --config configs/baseline.toml chunk --documents data/demo/documents.jsonl --output data/demo/passages/passages.jsonl
```

## Queries and gold evidence

Query JSONL rows contain `query_id`, `text` and optional `language` (`en`, `hi`,
`hinglish`). Each query must have exactly one gold record. `answerable` is required.

```json
{"query_id":"q1","answerable":true,"evidence_spans":[{"document_id":"wheat-source-001","content_sha256":"<original file SHA-256>","canonical_sha256":"<canonical text SHA-256>","start_char":120,"end_char":180}]}
{"query_id":"q2","answerable":false}
```

Hash placeholders above must be replaced with real 64-character hashes from the
document manifest. The annotator must select the correct canonical spans; the
system does not invent research labels. End offsets are exclusive.

A future manually judged passage-level record can instead contain
`relevant_passage_ids` and `chunking_config_id`, with `answerable: true`. Both
relevance types are supported, but cannot be mixed within one query. Use a uniform
label policy for comparable research runs. All positive labels are binary; graded
relevance is not implemented.

Missing records, malformed fields, duplicates, unknown IDs, out-of-bounds offsets,
changed fingerprints and incompatible passage-label configurations fail validation.
A well-formed gold span not coverable by any indexed passages generates an explicit
warning rather than silently changing the labels or denominator.

## Evaluation semantics

The initial policy is `strict_union`, recorded verbatim in every report:

- A gold span is covered only if the union of retrieved intervals from the same
  document and original/canonical versions covers **100% of its characters**.
  Whitespace counts. Adjacent or overlapping intervals can complete the evidence.
- Recall@K is covered distinct gold spans divided by all gold spans for the query.
  Retrieving multiple overlapping passages never counts one span multiple times.
- `cumulative_evidence_mrr_at_k` uses reciprocal rank of the earliest result prefix that fully covers any
  gold span. Partial evidence at rank 1 completed at rank 3 gives reciprocal rank
  1/3. No completion by K gives zero. This cumulative definition is explicit
  because it differs from single-passage relevance MRR.
- Passage-label mode uses conventional relevant-ID recall and first-relevant-hit
  reciprocal rank, reported as `mrr_at_k`. Each passage ID counts once.
- Metrics are macro-averaged across answerable queries. Explicitly unanswerable
  queries remain in input data and are reported with exclusion reasons. If every
  query is excluded, metrics are null, not zero.
- Both reciprocal-rank metrics are dictionaries keyed by cutoff, for example
  `"cumulative_evidence_mrr_at_k": {"1": 0.5, "5": 0.75}`. These numbers only
  illustrate the output format. The ambiguous `MRR@K` output key is not emitted.
  `mrr_at_k` is reserved for manually judged passage-level relevance. Each named
  metric averages only queries with its corresponding relevance type and is null
  at each cutoff if none are available; the two policies are never blended under
  one MRR label. Per-query `reciprocal_rank` values and scoring are unchanged.
  Set K at least as large as corpus size to obtain the corresponding full-ranking
  metric for BM25.

Run reports preserve rankings, query/label/corpus fingerprints, chunking IDs,
configuration, dependency versions, extraction versions and a deterministic run ID.
The creation timestamp is expected to differ across identical runs. Original source
paths may differ across machines without changing corpus identity.

## Testing and limitations

Tests exercise exact original preservation, page offsets, Unicode/CRLF handling,
empty input, deterministic chunk IDs and overlaps, section/page boundaries,
BM25 ranking and ties, strict interval coverage, metric arithmetic, explicit
unanswerable exclusions, bad-label rejection, alternate retrievers and the CLI.

Extraction quality for multicolumn PDFs, tables, figures, footnotes and scanned
pages is not guaranteed. No automated semantic section recognition or cross-page
continuity inference is performed. Corpus/index storage is local and in-memory,
suitable for a baseline scaffold rather than production-scale serving. Dependency
versions are pinned, but compatibility outside the tested host still needs checking.

## Recommended next milestone

**Corpus and annotation pilot:** approve a small set of English wheat-management
sources for India, ingest those approved local documents, audit extraction and
section boundaries, and manually annotate a versioned English/Hindi/Hinglish query
set with canonical evidence spans and explicit unanswerable labels. Establish
annotation and adjudication rules before reporting research metrics or adding dense
retrieval experiments. No real corpus acquisition is authorized by this scaffold.
