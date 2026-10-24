import pytest
from krishisaarthi.config import ChunkingConfig
from krishisaarthi.preprocessing.chunking import DeterministicChunker
from krishisaarthi.preprocessing.tokenization import UnicodeTokenizer
from krishisaarthi.ingestion.plaintext import PlainTextIngestor
from krishisaarthi.ingestion.pdf import PDFIngestor
from krishisaarthi.schemas import SourceMetadata
from conftest import FIXTURES


def test_unicode_tokens_keep_combining_marks():
    tokenizer = UnicodeTokenizer()
    assert tokenizer.terms('गेहूँ की खेती Wheat!') == ['गेहूँ', 'की', 'खेती', 'wheat', '!']
    text = 'गेहूँ wheat'
    assert all(text[t.start:t.end] == t.text for t in tokenizer.token_spans(text))


def test_deterministic_offsets_and_sections(document):
    chunker = DeterministicChunker(ChunkingConfig(size=12, overlap=3))
    first = chunker.chunk(document)
    assert first == chunker.chunk(document)
    assert len({p.passage_id for p in first}) == len(first)
    for p in first:
        assert p.text == document.canonical_text[p.evidence_span.start_char:p.evidence_span.end_char]
        assert p.token_count <= 12
        assert p.text.count('# ') <= 1
    assert {p.passage_id for p in first}.isdisjoint(p.passage_id for p in DeterministicChunker(ChunkingConfig(size=13, overlap=3)).chunk(document))


@pytest.mark.parametrize('size,overlap', [(1, 0), (4, 0), (4, 1), (4, 3), (256, 48)])
def test_window_overlap_and_complete_coverage(tmp_path, size, overlap):
    path = tmp_path / 'words.txt'
    path.write_text('  ' + ' '.join(f'word{i}' for i in range(17)) + '\n')
    doc = PlainTextIngestor().ingest(path, SourceMetadata())
    chunks = DeterministicChunker(ChunkingConfig(size=size, overlap=overlap)).chunk(doc)
    covered = set()
    for p in chunks:
        covered.update(range(p.evidence_span.start_char, p.evidence_span.end_char))
    assert covered == set(range(len(doc.canonical_text)))
    tokenizer = UnicodeTokenizer()
    for a, b in zip(chunks, chunks[1:]):
        left, right = tokenizer.terms(a.text), tokenizer.terms(b.text)
        assert len(set(left) & set(right)) == overlap


def test_page_bounded():
    doc = PDFIngestor().ingest(FIXTURES / 'synthetic_wheat.pdf', SourceMetadata())
    for passage in DeterministicChunker(ChunkingConfig()).chunk(doc):
        assert len(passage.page_references) == 1
        page = passage.page_references[0]
        assert page.canonical_start <= passage.evidence_span.start_char < passage.evidence_span.end_char <= page.canonical_end


@pytest.mark.parametrize('kwargs', [{'size': 0}, {'size': 4, 'overlap': 4}, {'overlap': -1}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)
