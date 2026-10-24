from pathlib import Path
from ..schemas import SourceMetadata
from .storage import make_document


class PlainTextIngestor:
    def ingest(self, path: Path, metadata: SourceMetadata):
        # Decode bytes directly: do not silently normalize CRLF or strip BOMs.
        text = path.read_bytes().decode("utf-8")
        if not text.strip():
            raise ValueError("Plain-text document is empty")
        return make_document(path, metadata, [(None, text)], "text/plain", "utf-8", "1")
