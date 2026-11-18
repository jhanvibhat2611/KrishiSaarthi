# Submission and demonstration guide

## Project scope

KrishiSaarthi retrieves English wheat evidence for English, Hindi and Hinglish
questions. Its primary application uses pinned multilingual E5 and shows the top
five passages with document/page references and canonical offsets. These are
retrieval results, not generated answers or agricultural recommendations.

The working application contains 15 manually selected PAU passages from Punjab's
Rabi 2025–26 package of practices. Source conditions and publication dates matter;
Punjab evidence must not be presented as universally applicable. The target of
approximately 100 useful passages is future collection work, not the current count.

## Run the existing local application

From the repository root on the configured development machine:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m krishisaarthi.ui.server --real --port 8766
```

Open http://127.0.0.1:8766. Stop with Ctrl-C. If that address is already serving
the app, use the running server. Startup validates frozen source provenance, the
pinned local E5 checkpoint and the existing passage embedding cache. Missing or
changed artifacts cause an error; the application does not silently download,
rebuild or switch models. The current loader deliberately accepts only Pilot v0.1.

A public checkout has code and synthetic fixtures, not the copyrighted corpus or
model weights. For a reproducible synthetic demonstration, install the baseline
dependencies as described in README, then run without `--real`:

```sh
PYTHONPATH=src python -m krishisaarthi.ui.server --port 8766
```

Choose **Synthetic demo fallback** explicitly. This verifies the interaction, not
the quality of real multilingual retrieval. Keep this distinction in the demo.

## Five-minute demonstration

1. Show that the real local E5 collection is available.
2. Submit one English question, its Hindi equivalent and a natural Hinglish variant.
   For example, ask about differences in the descriptions of Karnal bunt and loose
   smut. These are demonstration prompts, not new labeled evaluation records.
3. Expand citation details: passage ID, source document, physical page and offsets.
   Explain that English source text is retained rather than automatically translated.
4. Inspect whether the passages actually address both parts of the question; Top-5
   retrieval is not a guarantee of complete evidence or answerability.
5. Show the explicit synthetic fallback and error/empty-input behavior separately.

Do not describe similarity rankings as confidence in agricultural correctness.
Do not open or distribute raw local source documents as part of a public bundle.

## Architecture to explain

PDF/text ingestion preserves canonical text and provenance. The corpus workflow
admits human-reviewed regions and applies deterministic 256-token chunking with
48-token overlap. A Retriever interface supports BM25, E5, BGE-M3 and RRF. The app
uses existing E5 embeddings and encodes only the new question. The local Python
HTTP server passes trusted evidence metadata to the existing browser UI.

Evaluation is separate from live inference. Signed pilot queries and labels do
not enter the live retriever or model prompts. Recall measures covered evidence
spans; cumulative evidence MRR measures the first rank prefix covering any span;
complete evidence coverage requires all spans. Explicitly unanswerable queries
are excluded from positive-evidence metrics and counted separately.

## Dataset and experiment claims

| Material | Status and proper use |
| --- | --- |
| PAU Pilot v0.1 | 15 approved passages from one source; current live corpus |
| Original retrieval evaluation | 16 information needs, 48 signed language variants; exploratory pilot only |
| Other collected wheat documents | Review candidates or HOLD material; not automatically admitted |
| Code-mixing variants | Pending review; do not describe as a verified independent test set |
| Direct-versus-RAG outputs | Separate experimental outputs; human assessment remains incomplete |
| Synthetic fixtures | Engineering tests only, never research performance evidence |

Known Qwen failures include invalid structured output, unsupported claims and
unreliable supported Hindi/Hinglish answers. Citation structure alone cannot
establish factual support. Keep generation out of the main evidence-search demo.

## Completion gates for the larger collection

1. Review additional contiguous wheat source regions, preserving conditions,
   dates, geography, exact offsets and page identity. Leave existing HOLD decisions
   intact unless the responsible human explicitly resolves them.
2. Check duplicates, tables, privacy, extraction and reuse restrictions. Do not
   split or duplicate passages simply to reach 100; report actual distinct coverage.
3. Build a separately versioned approved snapshot with the existing corpus workflow.
   Preserve Pilot v0.1 and all historical experiments unchanged.
4. Prepare E5 embeddings bound to the new approved snapshot, then extend the app's
   validated loader to select it explicitly. This is not implemented by changing
   a passage-count constant or pointing the existing cache at different text.
5. Run provenance checks and local three-language application smoke tests. Existing
   pilot metrics do not transfer to a larger corpus without a separate evaluation.
6. Record the final collection size, source/topic coverage and unresolved gaps in
   the assignment report. Polish the UI after the functional collection is ready.

Source review is the current gate. Deferring generation assessment does not
authorize automatic source approval or answer-quality claims.

## Tests and handover

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q
node --test tests/ui.test.cjs
```

The HTTP tests require localhost socket access. `tests/qa_review.test.cjs` also
requires Playwright resolvable by Node and the installed Chrome path declared in
that file. It tests synthetic review data, not the real model. Do not download new
models or browsers simply to run the submission demo.

Keep `data/`, model weights, embeddings, annotations and experiment outputs out of
Git. Share source, dependency lockfiles, tests and documentation. Provide research
artifacts only through separately authorized local access. Human assessment may
remain deferred, but its incompleteness must remain explicit in the report.
