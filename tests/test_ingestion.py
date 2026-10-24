import hashlib
import pytest
from pypdf import PdfWriter
from krishisaarthi.schemas import SourceMetadata
from krishisaarthi.ingestion.pdf import PDFIngestor
from krishisaarthi.ingestion.plaintext import PlainTextIngestor
from krishisaarthi.ingestion.storage import save_document, write_jsonl, read_jsonl, PAGE_SEPARATOR
from krishisaarthi.schemas import Document
from conftest import FIXTURES


def test_pdf_pages_and_offsets():
    doc = PDFIngestor().ingest(FIXTURES / 'synthetic_wheat.pdf', SourceMetadata(source_url='https://example.invalid/test'))
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert 'inspectmarker' in doc.canonical_text[doc.pages[0].canonical_start:doc.pages[0].canonical_end]
    assert 'harvestmarker' in doc.canonical_text[doc.pages[1].canonical_start:doc.pages[1].canonical_end]
    assert doc.canonical_text[doc.pages[0].canonical_end:doc.pages[1].canonical_start] == PAGE_SEPARATOR
    assert doc.source_url == 'https://example.invalid/test'


def test_preserve_original_and_extracted(document, tmp_path):
    original = open(document.original_file_path, 'rb').read()
    saved = save_document(document, tmp_path)
    assert open(saved.original_file_path, 'rb').read() == original
    assert hashlib.sha256(original).hexdigest() == saved.content_sha256
    assert open(saved.pages[0].extracted_text_path, 'rb').read().decode() == document.canonical_text
    write_jsonl(tmp_path / 'docs.jsonl', [saved])
    assert read_jsonl(tmp_path / 'docs.jsonl', Document) == [saved]


def test_text_unicode_crlf(tmp_path):
    path = tmp_path / 'test.txt'
    text = 'गेहूँ wheat\r\nखेत water'
    path.write_bytes(text.encode())
    doc = PlainTextIngestor().ingest(path, SourceMetadata())
    assert doc.canonical_text == text
    assert doc.pages[0].page_number is None
    assert doc.pages[0].canonical_end == len(text)


def test_empty_pdf_rejected(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    path = tmp_path / 'empty.pdf'
    writer.write(path)
    with pytest.raises(ValueError, match='OCR'):
        PDFIngestor().ingest(path, SourceMetadata())


def test_empty_text_rejected(tmp_path):
    path = tmp_path / 'empty.txt'
    path.write_text('   ')
    with pytest.raises(ValueError, match='empty'):
        PlainTextIngestor().ingest(path, SourceMetadata())


def test_original_changed_rejected(document, tmp_path):
    path = tmp_path / 'changed.txt'
    path.write_text('different')
    changed = document.model_copy(update={'original_file_path': str(path)})
    with pytest.raises(ValueError, match='changed'):
        save_document(changed, tmp_path / 'saved')
