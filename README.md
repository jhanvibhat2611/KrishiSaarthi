# KrishiSaarthi

A research-oriented English/Hindi/Hinglish agricultural retrieval project with a
conservative, local evidence-grounded RAG scaffold. Current source scope is wheat;
the frozen pilot contains Punjab-specific, source-dated English evidence.

## Current implementation

- Text PDF/plain-text ingestion, canonical offsets, hashes and source/page provenance.
- Registry, page audits, reviewed-region selection and deterministic chunking
  (default 256 baseline Unicode tokens, 48-token overlap).
- BM25, multilingual E5-base and dense-only BGE-M3 through one Retriever interface.
- Equal-weight RRF and a separately labeled non-positive-BM25-gated diagnostic.
- Recall, explicitly named cumulative evidence MRR, ALL-span complete evidence
  coverage, per-language metrics and BM25 lexical diagnostics.
- RAG EvidenceSelector, conservative EvidenceSufficiencyChecker, PromptBuilder,
  Generator interface, deterministic mock, CitationValidator and RAGPipeline.
- Optional pinned local MLX adapter with offline checkpoint validation and measured local inference support.

No training, query translation/normalization, frontend, voice or image analysis is
implemented. Historical retrieval experiments remain separate from RAG. Public
source code/tests contain synthetic fixtures; research documents, extracted text,
annotations, weights and experiment outputs remain local under ignored `data/`.
No synthetic result is a research finding.

## Setup and tests

Python 3.11+; current development environment uses Python 3.12 on Apple Silicon.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
python -O -m pytest tests/test_runtime_validation.py tests/test_rag.py tests/test_mlx_adapter.py tests/test_page_provenance.py -q
```

For dense runs use `requirements-dense.lock` instead of `requirements.lock`.
Optional MLX/model assets require explicit approval and a separate environment; see docs/rag.md.
All runtime integrity checks use explicit exceptions and remain active under `-O`.
Pytest may warn about ordinary assertions under `-O`; subprocess guard tests verify
actual exceptions independently of test assertions. The full normal-mode suite is
still required.

## Synthetic ingestion and retrieval

Run commands from the repository root, using fresh local output paths:

```sh
python -m krishisaarthi.cli --config configs/baseline.toml ingest --manifest tests/fixtures/manifest.jsonl --output-dir data/demo-new
python -m krishisaarthi.cli --config configs/baseline.toml chunk --documents data/demo-new/documents.jsonl --output data/demo-new/passages.jsonl
python -m krishisaarthi.cli search --passages data/demo-new/passages.jsonl --query 'wheat' --k 3
python scripts/demo_rag.py
```

The last command uses a clearly synthetic review and mock generator; it returns a
`review_required` draft, never a factually certified answer or real human approval.

## Existing local corpus and experiments

```sh
python -m krishisaarthi.corpus --help
python scripts/run_dense_pilot.py --model e5 --output data/experiments/e5-pilot-v0.1-run-002
python scripts/run_dense_pilot.py --model bge-m3 --output data/experiments/bge-m3-pilot-v0.1-run-002
python scripts/run_hybrid_pilot.py --constant 60 --output data/experiments/hybrid-comparison-v0.1-run-002
```

Dense commands require the approved pinned checkpoints and validated frozen local
corpus/signed export. Run one model at a time. Hybrid replays full validated saved
baseline rankings. These commands never re-export annotations or change prior runs.
They are not runnable from public Git alone without authorized local artifacts.

## Conservative RAG CLI

```sh
python -m krishisaarthi.rag --documents data/pilot-v0.1/documents.jsonl --passages data/pilot-v0.1/passages.jsonl --question 'What source evidence is available about wheat soil suitability?' --evidence-only
python -m krishisaarthi.rag --documents data/pilot-v0.1/documents.jsonl --passages data/pilot-v0.1/passages.jsonl --question 'What source evidence is available about wheat soil suitability?' --generator mock
```

Without a new question-specific sufficiency review, the second command correctly
returns `insufficient_evidence` and does not call the generator. No gold labels or
annotation-derived answerability are consulted. An actual human may review the
first command's evidence pack and fill the separate support-review schema, then
pass `--review data/<new-review>.json`. This is not the research annotation sign-off
form, and the existing signed evaluation dataset must not be converted into it.

For explicitly controlled generation research, add `--mode research_draft`.
This mode attempts candidate generation from validated nonempty retrieved evidence
without inventing a sufficiency review. It keeps answerability unestablished and
returns drafts as `review_required`; abstentions and invalid outputs remain explicit.
The default `conservative` behavior is unchanged. Never use this mode as verified
agricultural advice. It does not consult gold labels.

Structural citation validation checks IDs, offsets and quoted source bytes. It
**does not prove factual entailment**. Every generated claim remains an unverified
draft requiring review, even after evidence has been judged sufficient to attempt.

## Architecture and documentation

`ingestion/` → `preprocessing/` → `retrieval/`; `evaluation/` independently scores
retrieved evidence. `corpus/` manages local provenance and admission. `rag/` composes
retrieval and generation without importing gold schemas or reading annotations.
Source-byte corpus validation is reused from the evaluation validation utility.

- [Corpus preparation and historical review workflow](docs/corpus-preparation.md)
- [Dense setup, revisions and reproduction](docs/dense-baselines.md)
- [RRF definitions and reproduction](docs/rrf-baselines.txt)
- [RAG contracts, manual sufficiency gate and optional MLX](docs/rag.md)

Recall counts covered gold spans once; cumulative evidence MRR finds the earliest
prefix covering any gold span; complete coverage requires all spans. Conventional
passage-level `mrr_at_k` is reserved for manually judged passage relevance.
Explicitly unanswerable queries remain in datasets but are excluded from positive
retrieval metrics. No score threshold constitutes a universal answerability test.

## Research integrity and limitations

The 15-passage single-document pilot is exploratory. Source matching and human
annotation sign-off do not certify independent agricultural correctness or current
recommendation validity. Preserve geographic, temporal, cultivation and numeric
conditions. Do not infer chemical recommendations or diagnosis from missing evidence.

Raw/restricted artifacts must not be force-added to Git. No commit/push is automatic.
The checkpoint includes source, configurations, dependency pins and synthetic tests;
new questions require genuine review before export or labeled evaluation.
