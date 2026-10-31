# Phase 2: dense-only multilingual baselines

The adapters implement the existing `index(passages)` / `search(query, k)` interface.
No BM25/evaluation scoring code is changed. No translation, query normalization,
sparse/ColBERT scoring, reranking, hybrid retrieval, fine-tuning or generation is used.

## Official recipes and pinned assets

- [Multilingual E5 base](https://huggingface.co/intfloat/multilingual-e5-base):
  `query: ` / `passage: ` prefixes for every language; attention-mask mean pooling,
  768 dimensions; 512-token limit including prefix and special tokens.
- [BGE-M3](https://huggingface.co/BAAI/bge-m3): no query instruction; CLS pooling,
  1024 dimensions; 8192-token limit. The model card permits Transformers dense
  inference. Its [official pooling recipe](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/baai_general_embedding)
  uses the first hidden-state token. Sparse and ColBERT heads are not downloaded.

`configs/dense-models.json` records exact model and tokenizer commit revisions.
The downloader is restricted to these two model IDs and selected files. LFS weight
hashes are checked against Hub metadata; a local download manifest binds all files.
Inference is local-only, `trust_remote_code=False`. BGE-M3's pinned checkpoint is
provided as `pytorch_model.bin`; the pinned PyTorch/Transformers stack uses the
weights-only loader. No remote model code is executed.

## Setup and run sequentially

```sh
source .venv/bin/activate
python -m pip install -r requirements-dense.lock
python -m pip install -e . --no-deps
python scripts/download_dense_models.py e5
python scripts/download_dense_models.py bge-m3
python -m pytest -q
python scripts/run_dense_pilot.py --model e5 --output data/experiments/e5-pilot-v0.1-run-002
# Wait for process exit before starting the next model.
python scripts/run_dense_pilot.py --model bge-m3 --output data/experiments/bge-m3-pilot-v0.1-run-002
```

Outputs must be new directories; these example run numbers must be changed if they
already exist. The runner is specific to the signed Pilot Corpus v0.1 and its saved
BM25 export; it never re-exports or changes annotations. Both corpus and BM25 file
integrity ledgers, annotation/sign-off hashes, label coverage and checkpoint hashes
are validated before encoding. All weights, caches and research output stay under
Git-ignored `data/`. Prior BM25 package versions/manifests are not rewritten.

Use `--device cpu|mps|auto` and `--batch-size N` (default 2). Auto selects MPS only
when PyTorch reports it available. Float32, eager attention, seed 0, deterministic
algorithms and four CPU threads are recorded. An MPS runtime/unsupported-operation
failure restarts the entire process on CPU and records the reason. Embeddings from
different devices are never mixed; device is part of the cache identity. MPS support
is implemented, but a CPU-only run is not evidence of successful MPS execution.

Before encoding, all passages and all 48 queries are tokenized with the exact model
prefixes and special tokens, `truncation=False`. Any oversized input stops the run;
there is no silent truncation, split or corpus rewrite. Counts are checked again at
the model boundary. Embeddings must have the expected dimensions and finite,
nonzero values, then are L2-normalized. Scores are float32 cosine dot products;
exact ties sort by passage ID. Floating-point results can vary across hardware.

Passage cache identity includes ordered full passage records, corpus fingerprint,
model/tokenizer revision, checkpoint file hashes and inference/dependency settings.
The NumPy cache is loaded with `allow_pickle=False`; its file hash, dimensions and
normalization are validated. Query vectors are cached in memory by full query
record, not query ID alone. The evaluator can re-index from the passage cache.

## Reporting and limitations

Each run saves schema-2 evaluation results, all 15 ranked passages per query,
covered/missing evidence IDs at 1/3/5, token diagnostics, config, checkpoint hashes,
source-code hashes, package versions, corpus/evaluation-set fingerprints and runtime.
Unanswerable queries are excluded from positive-evidence metrics but retain rankings.
Dense cosine scores are not BM25 lexical-match diagnostics; the evaluator marks that
optional diagnostic unavailable. No unanswerability threshold is inferred.

Cold model loading, passage/query encoding, and ranking times are separate; cache
hits are identified. Runtime measurements are single-run observations, not controlled
hardware benchmarks. The saved `packages_freeze` may identify the editable project
by its Git HEAD; use manifest source hashes to identify the uncommitted implementation,
not that HEAD alone. Install the checked-out project separately from the dependency lock.

This is a 15-passage, one-document exploratory pilot, with 12 answerable and four
unanswerable needs in three languages. Do not treat translated variants as independent
samples or claim statistical significance. Multi-span IN-08 and IN-10 require ALL
spans. Preserve both first-evidence MRR and complete coverage in comparisons.

After both runs complete, generate a new read-only comparison report with:

```sh
python scripts/compare_dense_pilot.py --e5 data/experiments/e5-pilot-v0.1-run-001 --bge data/experiments/bge-m3-pilot-v0.1-run-001 --output data/experiments/dense-comparison-v0.1-run-001
```

This verifies all three run ledgers and equal corpus/evaluation fingerprints, then
compares paired query IDs. Reports distinguish Hindi ALL-evidence recoveries at each
K, Hinglish metric losses versus BM25, rank-1 disagreements (including unanswerable
queries, marked separately), and both gold ranks for each multi-span question.
