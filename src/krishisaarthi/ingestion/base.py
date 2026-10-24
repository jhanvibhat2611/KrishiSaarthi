from pathlib import Path
from typing import Protocol
from ..schemas import Document, SourceMetadata


class DocumentIngestor(Protocol):
    def ingest(self, path: Path, metadata: SourceMetadata) -> Document: ...
