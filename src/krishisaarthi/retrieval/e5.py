"""Multilingual E5 base: official query/passage prefixes and masked mean pooling."""
from .dense import DenseRetriever

class E5Retriever(DenseRetriever):
    def __init__(self, encoder, spec, corpus_fingerprint, cache_dir=None):
        if (spec['model_id'],spec['pooling'],spec['query_prefix'],spec['passage_prefix'],spec['dimension'],spec['max_tokens']) != ('intfloat/multilingual-e5-base','mean','query: ','passage: ',768,512):
            raise ValueError('E5 adapter requires the approved E5-base recipe')
        super().__init__(encoder,spec,corpus_fingerprint,cache_dir)
