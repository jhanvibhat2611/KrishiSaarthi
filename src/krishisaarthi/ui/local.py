"""Read-only frozen PAU/E5 access. No annotation or gold-label access."""
import hashlib
import json
import os
import threading
from pathlib import Path
import numpy as np
from ..config import fingerprint
from ..schemas import Document, Passage, Query
from ..ingestion.storage import read_jsonl
from ..evaluation.labels import validate_corpus
from ..retrieval.dense import HFEncoder, normalized
from ..retrieval.e5 import E5Retriever

CORPUS = 'a919560be507b862be119a236fd1d2501f8d7c32d8fe6ea4951114b205f8c88a'

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def ledger(directory):
    for name, expected in json.loads((directory/'integrity.json').read_text()).items():
        path = (directory/name).resolve()
        require(path.is_relative_to(directory.resolve()) and sha(path) == expected, 'Frozen file integrity mismatch')

class CachedE5(E5Retriever):
    def index(self, passages):
        ordered = sorted(passages, key=lambda p:p.passage_id)
        require(len(ordered) == 15 and len({p.passage_id for p in ordered}) == 15, 'Expected 15 unique frozen passages')
        self.check_tokens([p.passage_id for p in ordered], [p.text for p in ordered], 'passage')
        binding = {'model_fingerprint':self.model_fingerprint, 'corpus_fingerprint':self.corpus_fingerprint,
                   'passages':fingerprint([p.model_dump() for p in ordered])}
        key = fingerprint(binding)
        cache = self.cache_dir/key
        meta = json.loads((cache/'manifest.json').read_text())
        require(meta['binding'] == binding and meta['passage_ids'] == [p.passage_id for p in ordered], 'Cache provenance mismatch')
        require(sha(cache/'embeddings.npy') == meta['array_sha256'], 'Cache bytes mismatch')
        matrix = np.load(cache/'embeddings.npy', allow_pickle=False)
        normalized(matrix, len(ordered), self.spec['dimension'])
        require(np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5), 'Cache vectors are not normalized')
        self._passages, self._vectors = ordered, matrix
        self.cache_hit, self.cache_key = True, key

class LocalRetrieval:
    def __init__(self, root):
        root = Path(root).resolve()
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        corpus = root/'data/pilot-v0.1'
        ledger(corpus)
        manifest = json.loads((corpus/'manifest.json').read_text())
        require(manifest['corpus_fingerprint'] == CORPUS, 'Wrong corpus fingerprint')
        docs = read_jsonl(corpus/'documents.jsonl', Document)
        passages = read_jsonl(corpus/'passages.jsonl', Passage)
        self.documents, self.passages = validate_corpus(docs, passages)
        require(len(docs) == 1 and docs[0].document_id == 'pau-rabi-2025-26' and len(passages) == 15, 'Wrong pilot size or source')
        require(fingerprint([p.model_dump() for p in passages]) == manifest['passages_fingerprint'] == '757436adcc6bb0244f92daaac56888637bd679ae2bcbd4e8dc5c8d518ebb6125', 'Frozen passage collection mismatch')
        spec = json.loads((root/'configs/dense-models.json').read_text())['e5']
        directory = root/'data/models/e5'/spec['revision']
        downloaded = json.loads((directory/'download-manifest.json').read_text())
        require(downloaded['model'] == spec, 'Wrong checkpoint specification')
        for name, expected in downloaded['files_sha256'].items():
            path = (directory/name).resolve()
            require(path.is_relative_to(directory.resolve()) and sha(path) == expected, 'Checkpoint integrity mismatch')
        spec = {**spec, 'checkpoint_files_fingerprint':fingerprint(downloaded['files_sha256'])}
        encoder = HFEncoder(directory, spec, device='cpu', batch_size=2)
        self.retriever = CachedE5(encoder, spec, CORPUS, root/'data/embeddings')
        self.retriever.index(passages)  # Read-only override; never encodes passages.
        self.lock = threading.Lock()

    def search(self, payload):
        from .server import validate_payload
        question, language = validate_payload(payload)
        query = Query(query_id='local-request', text=question, language=language)
        with self.lock:
            try:
                hits = self.retriever.search(query, 5)
            finally:
                self.retriever._query_vectors.clear()  # No retained query cache/history.
                self.retriever.token_lengths.pop('query:local-request', None)
        results = []
        for rank, hit in enumerate(hits, 1):
            p = self.passages.get(hit.passage_id)
            require(p is not None and hit.document_id == p.document_id and hit.evidence_span == p.evidence_span and hit.rank == rank, 'Returned provenance mismatch')
            doc = self.documents[p.document_id]
            pages = [x.page_number for x in p.page_references]
            results.append({'rank':rank, 'passage_id':p.passage_id, 'document_id':p.document_id,
                'title':doc.original_filename, 'text':p.text, 'pages':pages, 'source_url':doc.source_url,
                'section':'Physical PDF pages: '+', '.join(map(str,pages)),
                'citation':f"{doc.original_filename} · PDF page(s) {', '.join(map(str,pages))} · {p.passage_id}",
                'evidence_span':p.evidence_span.model_dump(), 'metadata':{k:p.domain_metadata.get(k) for k in ('region_id','printed_page_label','applicability','conditions_and_limitations','context_references')},
                'document_metadata':doc.domain_metadata})
        return {'mode':'local_e5', 'live_available':True, 'corpus_fingerprint':CORPUS,
                'results':results, 'warning':'Punjab · Rabi 2025–26. Retrieved evidence is not a verified answer or current agricultural advice.'}
