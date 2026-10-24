from typing import Protocol, Sequence
import regex
from ..config import ChunkingConfig, fingerprint
from ..schemas import Document, EvidenceSpan, Passage
from .tokenization import UnicodeTokenizer, TOKENIZER_ID


class PassageChunker(Protocol):
    def chunk(self, document: Document) -> Sequence[Passage]: ...


class DeterministicChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.tokenizer = UnicodeTokenizer()
        self.config_id = fingerprint({"config": config.model_dump(), "tokenizer": TOKENIZER_ID,
                                      "boundary_policy": "page-and-explicit-heading-v1"})

    def chunk(self, document: Document) -> list[Passage]:
        passages = []
        for page in document.pages:
            page_text = document.canonical_text[page.canonical_start:page.canonical_end]
            starts = {0, len(page_text)}
            starts.update(m.start() for m in regex.finditer(self.config.heading_pattern, page_text, regex.MULTILINE))
            boundaries = sorted(starts)
            for left, right in zip(boundaries, boundaries[1:]):
                region = page_text[left:right]
                tokens = self.tokenizer.token_spans(region)
                for first in range(0, len(tokens), self.config.size - self.config.overlap):
                    stop = min(first + self.config.size, len(tokens))
                    start = 0 if first == 0 else tokens[first].start
                    end = len(region) if stop == len(tokens) else tokens[stop].start
                    a, b = page.canonical_start + left + start, page.canonical_start + left + end
                    span = EvidenceSpan(document_id=document.document_id, content_sha256=document.content_sha256,
                        canonical_sha256=document.canonical_sha256, start_char=a, end_char=b)
                    passage_id = "passage-" + fingerprint({"span": span.model_dump(), "chunking": self.config_id})
                    passages.append(Passage(passage_id=passage_id, document_id=document.document_id,
                        text=document.canonical_text[a:b], evidence_span=span, page_references=[page],
                        token_count=stop-first, chunking_config_id=self.config_id,
                        original_filename=document.original_filename, source_url=document.source_url,
                        domain_metadata=document.domain_metadata))
                    if stop == len(tokens):
                        break
        return passages
