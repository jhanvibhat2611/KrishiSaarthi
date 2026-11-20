"""Encode an explicit validated application snapshot using the existing local E5.

No downloads, research evaluation, annotations, or changes to the pilot cache.
"""
import argparse
import json
import os
import time
from pathlib import Path
from krishisaarthi.corpus.application import load, private_path
from krishisaarthi.retrieval.application_e5 import ApplicationE5
from krishisaarthi.ui.local import load_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    _, passages, manifest = load(args.collection)
    start = time.perf_counter()
    encoder, spec = load_encoder(root)
    cache = private_path(root/'data/application-embeddings')
    retriever = ApplicationE5(encoder, spec, manifest['corpus_fingerprint'], cache)
    retriever.index(passages)
    report = {'corpus_fingerprint':manifest['corpus_fingerprint'], 'model_fingerprint':retriever.model_fingerprint,
              'model':spec, 'inference':encoder.identity, 'passage_count':len(passages),
              'cache_key':retriever.cache_key, 'cache_hit':retriever.cache_hit,
              'elapsed_seconds':time.perf_counter()-start, 'timings':retriever.timings,
              'token_lengths':retriever.token_lengths, 'research_evaluation':False}
    target = cache/retriever.cache_key/'preparation.json'
    if not target.exists():
        with target.open('x') as f:json.dump(report,f,indent=2)
        target.chmod(0o600)
    print(json.dumps({k:report[k] for k in ('passage_count','cache_key','cache_hit','elapsed_seconds')}))


if __name__ == '__main__':
    main()
