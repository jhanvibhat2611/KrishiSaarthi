"""Deterministic synthetic RAG demonstration; never approves research evidence."""
from pathlib import Path
from krishisaarthi.config import ChunkingConfig
from krishisaarthi.schemas import SourceMetadata, Query
from krishisaarthi.ingestion.plaintext import PlainTextIngestor
from krishisaarthi.preprocessing.chunking import DeterministicChunker
from krishisaarthi.retrieval.bm25 import BM25Retriever
from krishisaarthi.rag import EvidenceSelector, MockGenerator, RAGPipeline
from krishisaarthi.rag.models import SufficiencyReview, ComponentReview, Citation


def main():
    path=Path(__file__).resolve().parents[1]/'tests/fixtures/synthetic_wheat.txt'
    doc=PlainTextIngestor().ingest(path,SourceMetadata(document_id='synthetic-rag-demo'))
    selector=EvidenceSelector(BM25Retriever(),[doc],DeterministicChunker(ChunkingConfig()).chunk(doc))
    question=Query(query_id='synthetic-demo',text='Show synthetic wheat fixture evidence.')
    pack=selector.select(question)
    review=SufficiencyReview(evidence_fingerprint=pack.fingerprint,reviewer_id='SYNTHETIC-DEMO-NOT-A-HUMAN',
        reviewed_at='2025-01-01T00:00:00+00:00',all_question_components_covered=True,applicability_checked=True,
        components=[ComponentReview(component='Synthetic source display',support=[Citation(passage_id=p.passage_id,
            start_char=p.evidence_span.start_char,end_char=p.evidence_span.end_char,quote=p.text)]) for p in pack.passages])
    print(RAGPipeline(selector,MockGenerator()).run(question,review=review).model_dump_json(indent=2))

if __name__=='__main__':main()
