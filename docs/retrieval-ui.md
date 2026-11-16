# Local evidence explorer

Run from the repository root using the existing dense environment (no new install
or model download is needed):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m krishisaarthi.ui.server --real --port 8766
```

Open http://127.0.0.1:8766. Stop with Control-C. Without `--real`, the server exposes
only the synthetic backend; select **Synthetic demo fallback** explicitly in the UI.
The fallback is never substituted after a real-retrieval failure. Synthetic example
buttons explicitly switch to that labeled collection. No answer generation is invoked
by the interface. The older deterministic synthetic `/api/answer` contract remains
available for its existing tests/clients, but the UI uses retrieval endpoints only.

## Frozen local access

`ui/local.py` verifies the pilot integrity ledger, fixed corpus fingerprint,
exact serialized passage fingerprint, original PDF bytes, canonical text, offsets,
page files, pinned E5 checkpoint hashes and embedding-cache binding/checksum.
Only the 15 frozen PAU passages are admitted. It does not read annotations or gold
labels. `CachedE5.index` only loads the existing matrix; a missing or incompatible
cache aborts startup, without passage encoding or cache creation.

The existing E5 adapter encodes new questions offline, using its official query
prefix and token-limit check (no truncation). CPU float32, batch size 2 and the
installed package versions match the saved E5 cache identity. Ranking remains
cosine with stable passage-ID tie-breaking; Top-K is fixed to 5. No retrieval
algorithm or research baseline was changed. Query embeddings are cleared after
each request. No query history, access log or response log is written.

POST `/api/retrieve` accepts `query` and `language` (`en`, `hi`, `hinglish`). It
returns passages with trusted document/passage IDs, physical pages, canonical spans,
source hashes, source URL, printed page metadata, publication date and applicability.
It returns no answer or confidence percentage. POST `/api/search` remains synthetic.
GET `/api/status` reports local model availability without returning source text.

The standard-library server binds exclusively to 127.0.0.1, validates the Host and
Origin, enables no CORS, uses no-store responses, and serves only three allowlisted
static assets. Research files cannot be downloaded through static paths. Restricted
text is returned only dynamically to the local UI; no source text is bundled in
assets. No dependencies, external fonts, analytics, tunnels or deployment were added.
This is a private single-user development server, not an authenticated multi-user
service. Other processes on the same computer can reach localhost.

## Behavior and limitations

The cream/green layout remains. Enter submits; Shift+Enter creates a new line.
Requests disable duplicate submission, show loading, and time out after 60 seconds.
Empty/error states and keyboard-expandable citation details are provided. Language
selection labels the question; retrieved English source text is not translated.
Dense retrieval normally returns five passages even for unrelated questions. Their
presence does not establish relevance, complete evidence, answerability or current
agricultural correctness. Punjab/Rabi 2025–26 and individual source exceptions apply.
Printed labels are shown only when recorded; physical PDF pages remain explicit.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q
node --test tests/ui.test.cjs
```

Step 1: 399 Python tests and 7 JavaScript controller tests passed. Live browser smoke
checks returned five real passages for English, Hindi and Hinglish questions. Enter
submission and a 390px mobile viewport were checked; no horizontal page overflow.
Controller tests cover loading, duplicate suppression, errors, empty results,
language payloads, citations and separation from generated answers. Cache tests
reject missing/tampered/misbound artifacts without rebuilding. HTTP tests cover
unavailable/failed retrieval, private error suppression and Host/Origin guards.
No labeled evaluation or generation experiment was run. Full assistive-technology
and physical-device audits remain outside these smoke checks.
