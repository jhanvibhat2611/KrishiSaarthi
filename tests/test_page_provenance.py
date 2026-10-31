"""Synthetic persisted-page relocation tests; never modify the research corpus."""
import copy
from pathlib import Path
import pytest
from krishisaarthi.ingestion.storage import save_document
from krishisaarthi.preprocessing.chunking import DeterministicChunker
from krishisaarthi.config import ChunkingConfig
from krishisaarthi.evaluation.labels import validate_corpus

@pytest.fixture
def relocated(document,tmp_path):
    doc=save_document(document,tmp_path/'frozen')
    chunks=DeterministicChunker(ChunkingConfig()).chunk(doc)
    chunks=copy.deepcopy(chunks)
    other=tmp_path/'earlier-page.txt'
    other.write_bytes(Path(doc.pages[0].extracted_text_path).read_bytes())
    for p in chunks:
        p.page_references[0].extracted_text_path=str(other)
    return doc,chunks,other

def test_identical_relocated_pages_accepted_without_mutation(relocated):
    doc,chunks,_=relocated
    before=(doc.model_dump_json(),[p.model_dump_json() for p in chunks])
    validate_corpus([doc],chunks)
    assert before==(doc.model_dump_json(),[p.model_dump_json() for p in chunks])

@pytest.mark.parametrize('mode',['different','missing','document_tamper','document_missing','page_number','offset','doc_id','source_hash','canonical_hash','source_file','one_null'])
def test_bad_provenance_rejected(relocated,mode):
    doc,chunks,other=relocated
    if mode=='different':other.write_text('different')
    elif mode=='missing':other.unlink()
    elif mode=='document_tamper':Path(doc.pages[0].extracted_text_path).write_text('tampered')
    elif mode=='document_missing':Path(doc.pages[0].extracted_text_path).unlink()
    elif mode=='page_number':chunks[0].page_references[0].page_number=99
    elif mode=='offset':chunks[0].page_references[0].canonical_end-=1
    elif mode=='doc_id':chunks[0].evidence_span.document_id='wrong'
    elif mode=='source_hash':chunks[0].evidence_span.content_sha256='f'*64
    elif mode=='canonical_hash':chunks[0].evidence_span.canonical_sha256='f'*64
    elif mode=='source_file':Path(doc.original_file_path).write_text('tampered original')
    elif mode=='one_null':chunks[0].page_references[0].extracted_text_path=None
    with pytest.raises(ValueError):validate_corpus([doc],chunks)

def test_same_declared_path_still_checks_content(relocated):
    doc,chunks,_=relocated
    for p in chunks:p.page_references[0].extracted_text_path=doc.pages[0].extracted_text_path
    validate_corpus([doc],chunks)
    Path(doc.pages[0].extracted_text_path).write_text('wrong')
    with pytest.raises(ValueError):validate_corpus([doc],chunks)
