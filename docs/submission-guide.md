# Submission and demonstration guide

## Project scope

KrishiSaarthi retrieves English crop evidence for English, Hindi and Hinglish
questions. Its primary application uses pinned multilingual E5 and shows the top
five passages with document/page references and canonical offsets. These are
retrieval results, not generated answers or agricultural recommendations.

The working application contains 124 passages across wheat, barley, spring maize,
chickpea and rapeseed–mustard from Punjab's Rabi 2025–26 package of practices.
The 15 inherited wheat passages retain their pilot selection; the 109 added
passages are AI-assisted source selections, not human-verified research evidence. Source conditions and publication dates matter;
Punjab evidence must not be presented as universally applicable. The approximately 100-passage application target has been exceeded; geographic and
source diversity remain limited.

## Run the existing local application

From the repository root on the configured development machine:

```sh
./scripts/start_local_app.sh
```

Open http://127.0.0.1:8767. Stop with Ctrl-C. If that address is already serving
the app, use the running server. Startup validates frozen source provenance, the
pinned local E5 checkpoint and the existing passage embedding cache. Missing or
changed artifacts cause an error; the application does not silently download,
rebuild or switch models. The launcher explicitly selects the application snapshot. The original pilot
remains available through the original `--real` command without `--collection`.

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
| Application Multicrop v2 | 124 passages, five crop groups, one source; AI-assisted local collection |
| PAU Pilot v0.1 | 15 approved passages; frozen research corpus preserved |
| Original retrieval evaluation | 16 information needs, 48 signed language variants; exploratory pilot only |
| Other collected wheat documents | Review candidates or HOLD material; not automatically admitted |
| Code-mixing variants | Pending review; do not describe as a verified independent test set |
| Direct-versus-RAG outputs | Separate experimental outputs; human assessment remains incomplete |
| Synthetic fixtures | Engineering tests only, never research performance evidence |

Known Qwen failures include invalid structured output, unsupported claims and
unreliable supported Hindi/Hinglish answers. Citation structure alone cannot
establish factual support. Keep generation out of the main evidence-search demo.

## Status and remaining work

The multi-crop application and source-bound E5 cache are implemented. It uses the
same 256-token/48-overlap chunking settings with explicit source-section boundaries.
The frozen research pilot, labels, HOLD decisions and prior results are unchanged.

The expanded collection is an application dataset, not an automatically approved
research corpus. Human assessment remains optional follow-up for the assignment
but necessary before claiming human-verified evidence or answer quality. No new
labeled evaluation has been run. Existing pilot scores do not describe performance
on the expanded collection. UI visual polish can follow independently.

See `docs/multicrop-application.md` for snapshot verification, the explicit launch
command, cache reproduction and limitations. Do not publish local source text.

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
