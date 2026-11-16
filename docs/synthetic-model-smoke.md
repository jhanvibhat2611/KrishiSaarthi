# Offline real-model synthetic smoke

This optional CLI exercises the existing cached Qwen checkpoint through
`MLXTextBackend`. It never reads real corpora, signed labels or source registries.
It is not wired into the web UI: the UI stays visibly deterministic/test-double.
No Direct LLM versus RAG benchmark is completed by this smoke.

Exactly six requests are frozen before model loading: a sample-label question and
an absent collection-date question in English, Hindi and Hinglish. Each uses the
same original synthetic soil notebook fixture. Request IDs are logging identifiers,
not inference rules; only the question, language and source text reach the model.
There is no exact-question or expected-answer allowlist in the model adapter.

The loader validates every pinned file hash before loading, preserves the 6 GiB
MLX allocation and 256 MiB cache limits, and loads one model. Each request makes
one decode call, sequentially: 256 output tokens, temperature 0, seed 0, existing
8192 context limit, prefill 256 and pinned chat template. No truncation, retries,
format repair, additional warm-up generation or prompt-tuning loop. Runtime or
resource failure stops the remaining calls. Existing EOS, output-cap and repetition
stops remain active. No integrated grammar decoder is claimed.

Output has status draft/insufficient/uncertain, claims with evidence handles, and
reason. The original `synthetic-claims-v1` contract requires draft claims plus an
empty reason. The new `synthetic-claims-v2` contract permits a draft reason as
unverified explanatory metadata: it is retained separately and never appended to
the answer or treated as evidence of support. Both versions require cited claims
for drafts and no claims plus a nonempty reason for abstention. All answer facts
belong in claims; structural validation cannot detect every factual assertion
hidden in a reason, so human support review is still required. Invalid JSON, schema failure, incorrect status, bad citations,
output exhaustion, repetition and runtime errors are recorded distinctly. Application
metadata resolves valid handles to source IDs and full synthetic-passage intervals.
These structural checks do not verify entailment or calibrated confidence.

Authorized invocation (not an automatic rerun):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
.venv-rag/bin/python scripts/run_synthetic_model_smoke.py \
  --model-dir data/models/qwen3-4b-mlx/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b \
  --contract-version synthetic-claims-v2 \
  --output data/<fresh-authorized-synthetic-run>
```

Output directories must be fresh. Prompts/settings/source and code hashes are saved
before loading. Outputs include raw responses, strict validation outcomes, per-call
runtime/tokens/memory and completion count. The CLI never exports gold labels or
certifies a model's insufficiency decision. Subsequent reruns require authorization.

Tests use fake decoding only:

```sh
.venv-rag/bin/python -m pytest tests/test_real_synthetic.py -q
```

The completed six-call run is a local engineering observation. Its JSON and contract
failures must remain visible; do not relax validation retroactively or count raw
plausible claims as accepted answers. Any later contract change needs a versioned,
separately authorized test. Real UI integration remains gated on reliable output
mapping and explicit authorization. General multilingual QA, multi-passage support,
independent factual review and a controlled Direct LLM versus RAG evaluation remain
outstanding. Existing research generation and data boundaries are unchanged.

Contract versions are recorded in future manifests, completion records and per-call
responses. V1 remains selectable with `--contract-version synthetic-claims-v1`.
The original smoke-v1 artifacts predate this explicit field and remain immutable;
their Hindi/Hinglish nonempty-reason failures are not retroactively accepted.
English malformed JSON remains invalid in both versions. Missing/unknown handles,
invalid status, unexpected answer fields and abstentions carrying claims remain
errors. Neither contract is a factual-entailment verifier. Confidence stays
`not_estimated`, and draft claims stay `pending_review`.

V2 rejects repeated JSON object keys at every nesting level before shared schema
parsing, including identical values and escaped spellings of the same key. These
are `invalid_json` failures; the raw output is preserved without repair. V2 also
enforces the prompt's maximum of two claims (`schema_failure` above that limit).
V1 keeps its historical parser behavior and three-claim schema limit; it is only
a compatibility option, not a recommendation for new runs. Neither the shared
historical parser nor the model backend is changed by these V2 checks.

A future repeat requires explicit approval: exactly the same six synthetic requests
(English/Hindi/Hinglish × supported/absent-date), same fixture, checkpoint, 256-token
cap, temperature 0 and seed 0, with V2 recorded explicitly. One sequential call per
request; no warm-up, retries or substantive repairs. No such repeat is authorized
by changing the contract or running fake-decoding tests.
