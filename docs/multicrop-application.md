# Multi-crop local application

The user expanded the assignment scope to at least three crops and authorized
independent completion. Application Multicrop v2 therefore uses a separate,
explicit **AI-assisted application selection policy**. No human review record,
research approval, gold label, family assignment, split, or HOLD decision was
changed. The original research loader and pilot remain available unchanged.

## Collection

124 passages: wheat 16, barley 24, spring maize 22, chickpea/gram 25, and
rapeseed–mustard 37. The 109 additional passages come from existing PAU sections;
no document or model download was needed. Rapeseed–mustard is a crop group whose
subtypes must remain distinct in interpreting the passages. Spring maize is not
interchangeable with kharif maize.

One official English publication supplies this collection: PAU Package of Practices
for Crops of Punjab, Rabi 2025–26. It is useful local evidence coverage, not a
comprehensive Indian agricultural dataset. Licenses/redistribution permission
remain unknown/not granted; all source text is private and Git-ignored.

Ambiguous tables, incomplete cross-page cultivar descriptions, duplicated
sidebars, general health-benefit claims, and selected multi-page treatment protocols
are excluded. Existing Assam, ICAR and Meghalaya candidates remain outside this
collection. Original research HOLD records remain unchanged. The application-only v2 resolves the sowing-prose context issue through a narrowed interval and mandatory separately cited regional qualification; it does not create a research approval.

AI inspection of rendered pages and canonical text supports source selection; it
is not independent agricultural verification. Historical quantities and chemical
recommendations must not be presented as current advice. The UI generates no answer.

## Launch

From the configured repository:

```sh
./scripts/start_local_app.sh
```

Open http://127.0.0.1:8767. `PORT=8768 ./scripts/start_local_app.sh` selects another
local port. Ctrl-C stops the server. The equivalent explicit command is:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m krishisaarthi.ui.server --real --collection data/application-multicrop-v2/final-snapshot --port 8767
```

The app binds to 127.0.0.1, checks Host/Origin, does not log queries, and never serves
the research directory as static files. Missing assets fail startup; no model
substitution or hidden synthetic fallback occurs. The local agriculture-inspired interface places crop/language search beside collection context on desktop and before it on mobile. Optional controls are collapsible. Actual source metadata remains visible.

Choose All crops or an explicit crop. Filtering preserves E5 ordering within the
chosen crop and returns up to five results. It is not query routing or a change to
any research ranking baseline. Retrieved passages remain English for all three
query languages. Dense retrieval may return passages even for unrelated questions;
result presence is not proof of answerability.

## Integrity and reproduction

`corpus/application.py` reconstructs passages from exact contiguous canonical
source intervals, rejects overlap with excluded spans, and checks source bytes,
canonical hashes, page files, physical pages, interval bounds and text hashes.
It uses the existing deterministic 256-token/48-overlap chunker with section
boundaries. Original pilot passages are inherited without rewriting their records.
The private selection plan and snapshot ledger bind the complete collection.

A new collection can be built only into a fresh Git-ignored directory using
`application.build(documents, inherited_passages, selection_plan, output)`.
This application function creates no research approval or gold records.

To reproduce the separately stored E5 cache from the existing private snapshot:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/prepare_application_cache.py --collection data/application-multicrop-v2/final-snapshot
```

This explicitly encodes passages, uses only the already installed pinned
`intfloat/multilingual-e5-base` checkpoint, and checks model-specific token limits
without truncation. The application-only embedding view prepends recorded crop, section and PAU
source context to exact source text; displayed text and canonical spans are unchanged.
This representation has its own model/cache fingerprint and does not change the
research E5 adapter or saved baseline scores. Token validation includes this context.
Official `passage: ` / `query: ` prefixes, normalized cosine
ranking, stable passage-ID ties, CPU float32 and batch size 2 are retained.
Cache bindings record model/inference fingerprints, corpus fingerprint, passage
order and array hash. The application loader only reads this cache; it never
rebuilds it during startup. Pilot embeddings and baseline outputs are untouched.

## Verification and limits

The completion run passed 497 Python tests and 16 JavaScript/browser tests. Fifteen
unlabeled local searches exercised five crop groups in English, Hindi and Hinglish;
all returned five crop-matching passages with source metadata. This is a functional
smoke check, not a relevance measurement or benchmark score.

No labeled evaluation, Qwen generation, claim certification or final annotation
approval was performed. The expanded corpus needs independent relevance labels
before reporting retrieval quality; original pilot metrics do not transfer.

The inspected smoke cases still show weaker Hinglish ranking: some first results
address seed treatment rather than seed rate, or a related cultivar rather than
sowing/harvesting timing. The app does not certify relevance and exposes source
passages for inspection. No accuracy percentage is inferred from these examples.

## Coverage correction (v2)

The original wheat pilot omitted a common sowing-time question. The separate
application snapshot now adds a contiguous sowing paragraph, with the exact
regional rust-management caution independently validated and always displayed
when that passage is returned. The two spans remain separate citations; they are
not rewritten into a single recommendation. The original research HOLD stays
unchanged. The simple mustard crop/time table was visually checked row by row
with its mixed-crop footnote before inclusion. Other ambiguous tables remain out.

Per-crop coverage notes now disclose omissions before searching. These are
collection-level notes, not a query-answerability classifier. No cosine cutoff,
gold-label lookup, translation or query-specific ranking rule has been added.
The app still returns the closest five matches even for unsupported requests.

Private source inspection and common-question results are in
`data/application-multicrop-v2/`; earlier snapshots and reports remain intact.

## Application-only Hinglish assistance

`ui/hinglish.py` supplies a versioned, deterministic vocabulary glossary for the
local multi-crop application. It replaces recognized whole words/phrases before
E5 query encoding. It does not change the encoder, passage vectors, source records,
research retrievers, original query text, or historical rankings. The original
15-passage research UI path leaves assistance disabled.

Assistance applies only when Hinglish is selected and the checkbox is enabled.
The response returns `query_transformation` with the original question, actual
search text, glossary version and replacements. The UI displays the search text;
uncheck assistance to compare the original wording. Neither form is logged.

This is partial lexical normalization, not fluent translation or answerability
verification. Unknown tokens and ambiguous standalone words such as `jau` remain
unchanged. Numbers, variety identifiers and units are retained; recognized
negation and logical words are mapped explicitly. Word boundaries prevent
substring replacement. No gold evidence, expected answers, query-specific rule,
new model, fuzzy matching or automatic crop-filter change is used.

Twelve local before/after engineering checks are saved privately under
`data/application-hinglish-v1/`. These development examples are not an independent
benchmark and do not establish general Hinglish accuracy. Complex sentences,
unlisted spellings and polysemous words still require user inspection. The output
may remain grammatically awkward because it is search text, not a translation.

### Everyday terminology expansion

`ui/search_terms.py` adds an independent, default-on application option for English
and Hinglish. Recognized terms such as `watered` / `watering` append `irrigation`
to the search text. The original sentence is retained verbatim after any separately
selected Hinglish glossary step. The response records the expansion version and
exact added terms; users can disable either option independently. Disable both
to search the original wording. Hindi and the research pilot path remain unchanged.

The bounded dictionary does not map ambiguous standalone `water`, `plant`, `cut`
or `feed`. It does not infer answers, recommended actions, crops or thresholds.
No re-ranking, source-ID rules, passage re-encoding, new model or gold-label lookup
is involved. Token-limit checks still run on the actual expanded query, without
truncation. Negated questions can still retrieve passages that do not establish
the requested exception; topic similarity is not answerability.

Twelve before/after local development checks are preserved under
`data/application-search-terms-v1/`. The barley screenshot query moves the irrigation
section from second to first. Tested maize/mustard watering queries also retrieve
irrigation first. These checks are not a held-out accuracy evaluation, and the
incomplete wheat irrigation coverage remains unchanged.

### Static-data capability notices

Collection-wide lack of live prices and weather forecasts stays visible when a
crop is selected. `ui/availability.py` detects a bounded set of English, Hindi and
Romanized live-data expressions in the original query and returns a specific
capability notice. It never removes or reorders evidence and never assigns a gold
answerability label. A missing notice does not establish evidence sufficiency;
source climate descriptions and general price discussions are not automatically
classified as live-data requests. These simple rules can miss unfamiliar wording.

The user's Batch 1 review (10 questions, 18 searches) is preserved privately as
user-reported exploratory observations under `data/application-manual-batch1-v1/`.
It is not a verified evaluation export. Citation visibility was checked by the
user; source-page correctness was not independently verified in that review.

### Post-exploratory application corrections

The six manual-testing batches are exploratory diagnostics, not gold labels. The
application-only vocabulary v2 adds plural watering and explicit spacing terms.
The Hinglish glossary maps `jau` only with the explicit barley filter; other
contexts retain it unchanged. Assistance stays optional and transformed wording
remains visible. No frozen snapshot, embeddings or research rankings are changed.

Crop/filter mismatch, selected out-of-collection crop/region mentions and
clinical/treatment wording produce bounded clarification notices. These literal
rules are incomplete, do not infer answerability, and never change the selected
filter. Source conditions and remaining concerns are now visible on each evidence
card rather than only inside citation details. Structurally valid citations are
not factual verification.

Use both assistance switches for the application demo; switch them off to inspect
unassisted E5 behavior. Multi-part questions can still miss evidence, even when it
exists in the collection. Inspect source conditions before combining passages.
Private before/after checks are stored under `data/application-fixes-v1/`; they
are development checks, not independent evaluation or accuracy estimates.


## UI checkpoint (21 November 2026)

The latest full validation passed 554 Python tests and 22 frontend/browser tests.
These are engineering checks, not agricultural accuracy measurements. The hero
image (`ui/static/field-hero.jpg`) is an original AI-generated decorative wheat
image, not a research photograph or corpus evidence. It is bundled locally; no
remote image requests or new public data routes are introduced.

Hindi queries receive Hindi response guidance, but original evidence remains
English. Two private Hindi summary drafts are pending human review and are not
used as approved answers. No generation or automatic review approval is enabled.
