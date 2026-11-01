"""Separate lexical control; original BM25 and chunking tokenization are unchanged."""
import regex
from ..config import fingerprint
from ..preprocessing.tokenization import UnicodeTokenizer, TOKENIZER_ID
from .bm25 import BM25Retriever

FILTERED_TOKENIZER_ID = 'unicode-words-containing-letter-or-number-v1'
_HAS_CONTENT = regex.compile(r'[\p{L}\p{N}]')


class PunctuationFilteredTokenizer:
    """Keep baseline tokens containing a Unicode letter or number, casefolded.

    Hyphens, decimal points and percent signs are separate baseline tokens and
    are removed. No joining, stemming, translation or normalization is performed.
    Combining marks attached to words remain; isolated marks/symbols do not.
    This tokenizer is for retrieval terms only, never canonical text or chunking.
    """
    def terms(self, text: str) -> list[str]:
        return [term for term in UnicodeTokenizer().terms(text) if _HAS_CONTENT.search(term)]


class PunctuationFilteredBM25Retriever(BM25Retriever):
    """Same BM25 scoring/rank/diagnostic implementation, separate tokenizer identity."""
    def __init__(self, config=None):
        super().__init__(config)
        self.tokenizer = PunctuationFilteredTokenizer()
        self.retriever_id = 'bm25-punctuation-filtered-' + fingerprint({
            'config': self.config.model_dump(), 'base_tokenizer': TOKENIZER_ID,
            'tokenizer': FILTERED_TOKENIZER_ID,
        })
