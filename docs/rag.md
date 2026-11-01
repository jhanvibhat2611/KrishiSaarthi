# Minimal local RAG: current contract

The default pipeline is deliberately **review-gated**, not an autonomous
answerability detector. Without a bound support review it returns insufficient
evidence. This is a functional conservative implementation; automatic semantic
sufficiency and factual verification remain unimplemented and uncalibrated.

## Modules

- `EvidenceSelector`: uses any existing Retriever, validates source-byte provenance,
  checks each result's query/rank/document/span, and retains metadata and pages.
- `EvidenceSufficiencyChecker`: requires nonempty evidence plus a fresh question/
  evidence-bound review covering every requested component, applicability and no
  unresolved concerns. No gold or score thresholds. It only authorizes an attempt.
- `PromptBuilder`: deterministic JSON source pack with request-local handles, dates,
  applicability metadata and constraints. The application retains exact source IDs
  and offsets; source/question text is untrusted data.
- `Generator`: interchangeable interface. `MockGenerator` returns clearly marked
  excerpts, not factual answers. `MLXGenerator` is optional and local-only.
- `CitationValidator`: checks provided passage IDs, canonical offsets and exact
  quotes, not whether those quotes entail a generated claim.
- `RAGPipeline`: combines these modules, suppresses invalid/failed/insufficient
  answers and marks every structurally valid generated draft `review_required`.

## Output

`original_question`, `retrieved_passage_ids`, `evidence_fingerprint`, `status`,
`answerability`, `generated_answer`, `claims` with citations, `sources`, `warnings`,
`factual_verification` and `diagnostics`. Sources preserve document metadata,
physical page references and passage metadata (including printed-page information
when available). The MLX handle response is resolved into application-owned citations. The generated answer is assembled only from structured claims;
there is no uncited free-form answer field from the generator.

Statuses: `insufficient_evidence`, `review_required`, `invalid_output`,
`generator_error`. There is intentionally no automatically certified `answered`
status. A valid citation can still accompany an unsupported claim. All claims are
flagged as unverified; no implemented verifier can identify every unsupported claim.

## Human sufficiency review

Use `python -m krishisaarthi.rag ... --evidence-only` to obtain the exact pack and
an incomplete template. Review the **new question**, all required components and
source restrictions; fill your actual identity and timestamp with timezone.
Each component needs a citation with `passage_id`, absolute `[start_char,end_char)`
offsets and exact `quote`. Set `all_question_components_covered` and
`applicability_checked` only after checking them, and retain unresolved concerns.
Pass the completed local JSON with `--review`. Changed question/evidence/order or
metadata invalidates its fingerprint. A timestamp/identity field is a user
attestation, not authenticated identity; never generate these fields on behalf of
a real reviewer. No converter from research gold or sign-off is provided.

For multi-part questions every requested component must be reviewed. A single
supported component is insufficient for a complete answer. After generation,
missing-context or abstention signals suppress the answer regardless of the earlier
review. Review generated claims separately for factual support.

## Optional MLX adapter and approved local inference

Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit`
Revision/tokenizer: `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`.
`configs/mlx-generator.json` pins file sizes and repository/LFS content checksums,
package versions and inference settings. The approved checkpoint can be stored in the ignored local cache. No checkpoint download is implicit in the adapter; it still requires all expected files locally.

After explicit approval for the model download and environment setup, use a separate
Apple Silicon environment with the optional `rag-mlx` extra (MLX 0.26.5, MLX-LM
0.26.3, Transformers 4.57.1). `requirements-rag.lock` records the resolved Apple Silicon runtime. Install it in
a separate environment, then install this project with `--no-deps -e .`.
Do not alter the retrieval experiment environment silently.

The adapter accepts only an existing local directory; every expected file must
match the pinned metadata before loading. The tokenizer uses local files only and
remote code is disabled. Prompt formatting uses the pinned chat template with
`tokenize=True` and `add_generation_prompt=True`. Greedy temperature-zero decoding,
seed 0, batch 1, prefill size 256, default 512 new tokens, 8192 total-token budget.
`--max-tokens` is configurable. Oversized prompts fail explicitly, never truncate.
Greedy decoding is not a promise of cross-hardware bitwise identity.

Once approved, use the same CLI with `--generator mlx --model-dir <local directory>`
and a completed request review. No download fallback exists. Actual model loading
and generation record elapsed time, MLX peak allocated bytes, macOS process-lifetime
peak RSS, token counts, revision and prompt-token hash. MLX peak bytes and RSS are
different measurements; neither measures whole-system peak usage. Unit tests use a fake MLX runtime. Actual local runs record measurements in
ignored reports, separately from simulated tests. A 6 GiB MLX allocation limit
and 256 MiB cache limit bound resource use; allocation failure stops the run.

Official references:
https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit
https://github.com/ml-explore/mlx-lm

## Restrictions and failure behavior

Preserve source dates, crop/region, units/product mass and all cultivation exceptions.
No automatic pesticide/fertilizer recommendation, diagnosis or current validity
claim. Missing evidence is corpus-relative; don't claim the information does not
exist elsewhere. Prompt instructions reduce risk but are not an entailment proof.
Retrieval/provenance errors fail closed with exceptions. Generator exceptions are
returned as `generator_error` with exception type only, avoiding private payloads.
Restricted source packs and demonstrations stay under ignored `data/`.

## Explicit research-draft mode

`RAGPipeline.run(..., mode="research_draft")` and CLI `--mode research_draft`
permit candidate generation without claiming semantic sufficiency. Provenance
validation, source checks, nonempty evidence, context limits and citation validation
remain mandatory. The missing human-review reason remains visible, alongside an
explicit research warning. `answerability` stays `not_established`. Generated
claims remain unverified; invalid citations suppress the candidate answer, while
generator abstention or reported missing context returns insufficient evidence.
No question ID, signed answer, gold span or annotation answerability is consulted.

For a reproducible local run, prepare a JSONL file containing Query records only:
query_id, text and optional language. Then use:

```sh
.venv-rag/bin/python scripts/run_local_rag_demo.py \
  --queries data/<questions-only>.jsonl \
  --model-dir data/models/qwen3-4b-mlx/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b \
  --output data/<fresh-generation-run> --max-tokens 768
```

The runner uses fixed BM25 Top-5 and all three question languages unchanged,
without translation or gold access. It saves prompts, full evidence, raw model
text (including malformed/invalid output), structured responses, checksums and
resource metrics locally. These are generation observations, not independent
human verification or a new retrieval baseline. Corpus-relative missing evidence
must not be conflated with absence from the full publication.

The prompt's compact output example reduces schema overhead without removing
source text or applicability conditions. A budget failure is explicit and must
be saved as a stopped attempt; retries use a fresh output directory. Report raw
outputs and failures, including unsupported claims despite structurally valid
citations. A model's own abstention/confidence is not an entailment verifier.

## Phase 4.1: application-managed evidence handles

The current prompt version is `source-grounded-handles-v1`. For each request,
retrieval order defines E1, E2, ...; the same evidence pack/order produces the same
handles. Handles are request-local, not document IDs, passage IDs or gold IDs.
The application retains complete validated passages, source hashes, canonical
intervals, page references and metadata. The MLX generator returns only:

```json
{"answer_status":"draft","answer_text":"Synthetic claim.","claims":[{"text":"Synthetic claim.","evidence_handles":["E1"]}],"uncertainty":[]}
```

`answer_text` must equal the claim texts joined with newline, preventing an
additional uncited answer field. An insufficient-evidence response requires empty
answer text and claims, with explicit uncertainty. The resolver rejects unknown,
malformed or repeated handles and constructs full-passage citations from trusted
metadata. The existing exact quote/offset validator still runs. Whole-passage
citation scope is structural provenance, not claim-level entailment or verified
minimal supporting evidence. The source text is never rewritten.

The installed MLX runtime is used with strict JSON/schema validation; no constrained
JSON decoder has been established here. This experiment permits **zero formatting
repair attempts** (within the maximum of one); it never silently repairs facts,
handles or quotations. `failure_category` distinguishes `model_runtime_error`,
`output_length_exhaustion`, `invalid_json`, `schema_failure`, `invalid_citation`
and `missing_evidence`. `status=review_required` identifies a candidate requiring
factual review. A generator failure is never counted as an abstention. The existing
broad status fields remain compatible; inspect the category for the exact failure.

To run the comparison offline, first prepare rankings in the baseline environment:

```sh
.venv/bin/python scripts/run_rag_reliability.py prepare \
  --queries data/<questions-only>.jsonl --output data/<fresh-retrieval-directory>
.venv-rag/bin/python scripts/run_rag_reliability.py generate \
  --queries data/<questions-only>.jsonl \
  --rankings data/<fresh-retrieval-directory>/rankings.json \
  --output data/<fresh-generation-directory> \
  --model-dir data/models/qwen3-4b-mlx/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
```

Preparation uses pinned local E5 on CPU, batch 2, then exits before Qwen loads.
Generation replays query/corpus-bound complete rankings through the Retriever
interface and validates selected provenance. Both arms use Top-5, the identical
prompt template, temperature 0, seed 0 and 768 output tokens; only evidence differs.
All selected query records contain questions only; neither stage reads gold labels.
The report may inspect evidence completeness afterwards, separately from inference.
Language quality and factual support require human review. Original experiments
remain unchanged; new output directories must be fresh and Git-ignored.

## Phase 4.2: optional candidate selection followed by short claims

`TwoStageGenerator` is a separate, opt-in research generator wrapping the existing
pinned MLX loader. The conservative pipeline's review gate is unchanged. Invoke
`scripts/run_rag_reliability.py generate --two-stage` with the same arguments and
a new ignored output directory to use it. Reuse the frozen ranking snapshot so
this comparison changes generation, not retrieval. No new model is downloaded.

Stage A emits `{status, handles, reason}` with candidate/insufficient/uncertain.
Candidate selection is a model proposal, never a verification result. Stage B sees
only selected passages, with their original request-local handles and applicability
metadata, and emits `{status, claims, reason}`. Each claim has text and evidence
handles. There is no repeated answer_text: the application assembles the response
from the claims. Stage B can still abstain or report uncertainty. Unknown or
unselected handles fail validation. Source IDs, full exact quotations and canonical
intervals remain application-owned and pass the existing citation validator.

Fixed budgets: 192 Stage-A tokens, 384 Stage-B tokens, temperature 0, seed 0, same
8192 context budget and pinned chat template. Stops are model EOS, output cap, or
six consecutive identical whitespace-token blocks of length 1–8. The repetition
guard is an engineering stop, not a faithfulness classifier. It may miss other
forms of repetition. No repairs or retries. Complete stage prompts, raw outputs,
selected handles, parsed records, termination reasons and resource metrics are
saved. Runtime/JSON/schema/citation/status/repetition/output-cap failures remain
separate. `uncertain` is retained in the stage record and response warning even
though the public response's conservative status is `insufficient_evidence`.

Inspection of pinned mlx-lm 0.26.3's generate.py and sample_utils.py finds general
logits processors, but no integrated JSON-schema or grammar-constrained decoder.
That hook alone does not establish a reliable JSON implementation. No grammar
library or major dependency was added. `constrained_json=True` explicitly raises
NotImplementedError instead of silently claiming constrained generation.

Phase 4.2 compares this fixed bundled design, not isolated causal effects of each
prompt/budget change. Stage selection and shortened generation are potential new
failure points. Claim-support and language assessments are assistant observations
for human review, not independent verification. If Hindi/Hinglish remain unreliable,
stop prompt tweaking and propose a separately approved controlled generator-model
comparison, retaining the same evidence and review rubric.
