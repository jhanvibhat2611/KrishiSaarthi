"""Shared dense-only inference/ranking. No remote inference, normalization or truncation."""
from krishisaarthi.validation import require
from pathlib import Path
import hashlib
import json
import time
import importlib.metadata
import numpy as np
from ..config import fingerprint
from ..schemas import RetrievalResult


class TokenLimitError(ValueError):
    pass


class DeviceFallbackRequired(RuntimeError):
    """Abort this device run; caller must restart the entire run on CPU."""


def normalized(array, rows, dimension):
    a = np.asarray(array, dtype=np.float32)
    if a.shape != (rows, dimension) or not np.isfinite(a).all():
        raise ValueError('Invalid dense embedding shape or nonfinite values')
    lengths = np.linalg.norm(a, axis=1, keepdims=True)
    if (lengths <= 0).any():
        raise ValueError('Zero dense embedding')
    return a / lengths


class HFEncoder:
    """Official Transformers pooling recipes; checkpoint/tokenizer loaded locally."""
    def __init__(self, directory, spec, device='auto', batch_size=2):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if device not in ('auto', 'mps', 'cpu') or batch_size < 1:
            raise ValueError('Invalid device or batch size')
        self.torch, self.spec, self.batch_size = torch, spec, batch_size
        self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'mps' and not torch.backends.mps.is_available():
            self.device = 'cpu'
        torch.manual_seed(0)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = 'right'
        self.model = AutoModel.from_pretrained(directory, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.float32, attn_implementation='eager')
        if self.model.config.hidden_size != spec['dimension']:
            raise ValueError('Checkpoint dimension mismatch')
        self.model.eval()
        try:
            self.model.to(self.device)
        except RuntimeError as exc:
            self._device_error(exc)
        self.identity = {'backend':'transformers-auto-model-v1','device':self.device,
            'dtype':'float32','attention':'eager','batch_size':batch_size,'seed':0,
            'deterministic_algorithms':True,'cpu_threads':4,'padding_side':'right',
            'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','numpy','safetensors','sentencepiece']}}

    def _device_error(self, exc):
        if self.device == 'mps' and any(s in str(exc).lower() for s in ('mps', 'out of memory', 'not implemented', 'deterministic')):
            raise DeviceFallbackRequired(str(exc)) from exc
        raise exc

    def token_counts(self, texts):
        encoded = self.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        return [len(x) for x in encoded['input_ids']]

    def encode(self, texts):
        torch = self.torch
        # Defensive second check at the actual model boundary.
        counts = self.token_counts(texts)
        if any(n > self.spec['max_tokens'] for n in counts):
            raise TokenLimitError('Input exceeds model limit; truncation prohibited')
        output = []
        try:
            for first in range(0, len(texts), self.batch_size):
                subset = texts[first:first+self.batch_size]
                batch = self.tokenizer(subset, add_special_tokens=True, padding=True,
                    truncation=False, return_tensors='pt')
                require(batch['attention_mask'].sum(1).tolist() == counts[first:first+self.batch_size], 'dense.py:81: integrity validation failed')
                batch = {k:v.to(self.device) for k,v in batch.items()}
                with torch.inference_mode():
                    hidden = self.model(**batch).last_hidden_state
                    if self.spec['pooling'] == 'mean':
                        mask = batch['attention_mask'][...,None].bool()
                        pooled = hidden.masked_fill(~mask, 0).sum(1) / mask.sum(1)
                    else:
                        pooled = hidden[:,0]
                    output.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
            if self.device == 'mps':
                torch.mps.synchronize()
        except (RuntimeError, NotImplementedError) as exc:
            self._device_error(exc)
        return np.concatenate(output, axis=0)


class DenseRetriever:
    def __init__(self, encoder, spec, corpus_fingerprint, cache_dir=None):
        self.encoder, self.spec = encoder, dict(spec)
        self.corpus_fingerprint = corpus_fingerprint
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_fingerprint = fingerprint({'spec':self.spec,'inference':encoder.identity})
        self.retriever_id = 'dense-cosine-' + self.model_fingerprint
        self._passages, self._vectors, self._query_vectors = [], None, {}
        self.token_lengths = {}
        self.timings = {'passage_encoding_seconds':0.,'query_encoding_seconds':0.,'ranking_seconds':0.,'token_validation_seconds':0.}
        self.cache_hit = False

    def formatted(self, texts, kind):
        return [self.spec[kind+'_prefix'] + t for t in texts]

    def check_tokens(self, ids, texts, kind):
        start = time.perf_counter()
        lengths = self.encoder.token_counts(self.formatted(texts,kind))
        self.timings['token_validation_seconds'] += time.perf_counter()-start
        if len(lengths) != len(texts):
            raise ValueError('Tokenizer length mismatch')
        for key,n in zip(ids,lengths):
            self.token_lengths[kind+':'+key] = {'tokens_with_prefix_and_specials':n,'limit':self.spec['max_tokens'],'within_limit':n<=self.spec['max_tokens']}
        excessive = [(key,n) for key,n in zip(ids,lengths) if n>self.spec['max_tokens']]
        if excessive:
            raise TokenLimitError(f'No truncation allowed: {excessive}')
        return lengths

    def preflight(self, passages, queries):
        self.check_tokens([p.passage_id for p in passages],[p.text for p in passages],'passage')
        self.check_tokens([q.query_id for q in queries],[q.text for q in queries],'query')

    def index(self, passages):
        if not passages or len({p.passage_id for p in passages}) != len(passages):
            raise ValueError('Nonempty unique passages required')
        ordered = sorted(passages,key=lambda p:p.passage_id)
        texts = [p.text for p in ordered]
        self.check_tokens([p.passage_id for p in ordered],texts,'passage')
        binding = {'model_fingerprint':self.model_fingerprint,'corpus_fingerprint':self.corpus_fingerprint,
                   'passages':fingerprint([p.model_dump() for p in ordered])}
        key = fingerprint(binding)
        cache = self.cache_dir/key if self.cache_dir else None
        if cache and cache.exists():
            meta = json.loads((cache/'manifest.json').read_text())
            if meta['binding'] != binding or hashlib.sha256((cache/'embeddings.npy').read_bytes()).hexdigest()!=meta['array_sha256']:
                raise ValueError('Dense cache fingerprint mismatch')
            matrix = np.load(cache/'embeddings.npy', allow_pickle=False)
            normalized(matrix,len(ordered),self.spec['dimension'])
            if not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-5):
                raise ValueError('Cached embeddings are not normalized')
            self.cache_hit = True
        else:
            start = time.perf_counter()
            matrix = normalized(self.encoder.encode(self.formatted(texts,'passage')),len(ordered),self.spec['dimension'])
            self.timings['passage_encoding_seconds'] += time.perf_counter()-start
            if cache:
                cache.mkdir(parents=True,exist_ok=False)
                np.save(cache/'embeddings.npy',matrix,allow_pickle=False)
                (cache/'manifest.json').write_text(json.dumps({'binding':binding,'array_sha256':hashlib.sha256((cache/'embeddings.npy').read_bytes()).hexdigest(),'shape':list(matrix.shape),'passage_ids':[p.passage_id for p in ordered]},indent=2)+'\n')
        self._passages,self._vectors=ordered,matrix
        self.cache_key=key

    def prepare_queries(self, queries):
        if not queries or len({q.query_id for q in queries})!=len(queries):
            raise ValueError('Nonempty unique queries required')
        self.check_tokens([q.query_id for q in queries],[q.text for q in queries],'query')
        start=time.perf_counter()
        matrix=normalized(self.encoder.encode(self.formatted([q.text for q in queries],'query')),len(queries),self.spec['dimension'])
        self.timings['query_encoding_seconds']+=time.perf_counter()-start
        for q,v in zip(queries,matrix):
            self._query_vectors[fingerprint(q.model_dump())]=v

    def search(self, query, k):
        if k<=0:raise ValueError('k must be positive')
        if self._vectors is None:raise ValueError('Index the corpus before searching')
        key=fingerprint(query.model_dump())
        if key not in self._query_vectors:self.prepare_queries([query])
        start=time.perf_counter()
        scores=self._vectors @ self._query_vectors[key]
        order=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),self._passages[i].passage_id))[:k]
        hits=[RetrievalResult(query_id=query.query_id,passage_id=self._passages[i].passage_id,
            document_id=self._passages[i].document_id,rank=rank,score=float(scores[i]),
            evidence_span=self._passages[i].evidence_span,retriever_id=self.retriever_id) for rank,i in enumerate(order,1)]
        self.timings['ranking_seconds']+=time.perf_counter()-start
        return hits
