from pathlib import Path
import pypdf
from ..schemas import SourceMetadata
from .storage import make_document


class PDFIngestor:
    def ingest(self, path: Path, metadata: SourceMetadata):
        reader = pypdf.PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are unsupported")
        pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        if not any(text.strip() for _, text in pages):
            raise ValueError("PDF has no extractable text; OCR is outside Phase 1")
        doc = make_document(path, metadata, pages, "application/pdf", "pypdf", pypdf.__version__)
        doc.extraction_metadata["empty_pages"] = [n for n, text in pages if not text.strip()]
        return doc
