from pathlib import Path
import pytest
from krishisaarthi.schemas import SourceMetadata
from krishisaarthi.ingestion.plaintext import PlainTextIngestor
from krishisaarthi.preprocessing.chunking import DeterministicChunker
from krishisaarthi.config import ChunkingConfig

FIXTURES = Path(__file__).parent / 'fixtures'

@pytest.fixture
def document():
    return PlainTextIngestor().ingest(FIXTURES / 'synthetic_wheat.txt', SourceMetadata(document_id='synthetic-text'))

@pytest.fixture
def passages(document):
    return DeterministicChunker(ChunkingConfig()).chunk(document)
