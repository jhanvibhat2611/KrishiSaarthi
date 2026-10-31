"""BGE-M3 dense-only CLS embeddings; no sparse or ColBERT heads/instructions."""
from .dense import DenseRetriever

class BGEM3Retriever(DenseRetriever):
    def __init__(self, encoder, spec, corpus_fingerprint, cache_dir=None):
        if (spec['model_id'],spec['pooling'],spec['query_prefix'],spec['passage_prefix'],spec['dimension'],spec['max_tokens']) != ('BAAI/bge-m3','cls','','',1024,8192):
            raise ValueError('BGE-M3 adapter requires the approved dense-only recipe')
        super().__init__(encoder,spec,corpus_fingerprint,cache_dir)
