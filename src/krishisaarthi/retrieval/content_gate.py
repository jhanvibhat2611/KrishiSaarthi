"""Minimal diagnostic lexical mask using query/candidate text only, never labels."""
import math
from ..config import fingerprint
from .punctuation_filtered import PunctuationFilteredTokenizer
from .rrf import RRFRetriever

# Fixed before this run; no pilot-label or per-query tuning. Exact matches only.
FUNCTION_WORDS = frozenset('a an the and or but of to in on at for from with by as is are was were be been being it its this that these those what which when where how why who whom whose do does did has have had can could may might must shall should will would not no all any some each both than then if so such into about after before during under over between through per also'.split())
GENERIC_WORDS = frozenset('wheat crop crops field fields farm farming agriculture agricultural'.split())
CONTENT_POLICY_ID = 'filtered-exact-content-and-numbers-v1'


class ContentTokenPolicy:
    """Keep exact filtered terms except fixed function/generic words.

    Numeric/alphanumeric tokens, units and named-entity words are not removed.
    No entity lexicon, translation, semantic equivalence or quantity reasoning.
    """
    def __init__(self):
        self.tokenizer = PunctuationFilteredTokenizer()
        self.identity = {'policy': CONTENT_POLICY_ID,
                         'function_words': sorted(FUNCTION_WORDS),
                         'generic_words': sorted(GENERIC_WORDS),
                         'numbers': 'retained as baseline tokens; no quantity interpretation'}

    def terms(self, text):
        return set(self.tokenizer.terms(text)) - FUNCTION_WORDS - GENERIC_WORDS

    def shared(self, query, passage):
        return sorted(self.terms(query) & self.terms(passage))


class _ContentMask:
    """Internal RRF input: retain ranks/provenance; zero masked contributions.

    Not a standalone scoring retriever. The underlying scores/results are untouched.
    RRF checks every result's provenance, including candidates that will be gated.
    """
    def __init__(self, lexical, policy):
        self.lexical = lexical
        self.policy = policy
        self.retriever_id = 'content-mask-' + fingerprint({
            'lexical': lexical.retriever_id, 'policy': policy.identity})
        self.passages = None

    def index(self, passages):
        self.passages = {p.passage_id: p for p in passages}
        self.lexical.index(passages)

    def decisions(self, query, k):
        if self.passages is None:
            raise ValueError('Index first')
        rows = []
        for hit in self.lexical.search(query, k):
            if hit.passage_id not in self.passages or not math.isfinite(hit.score):
                raise ValueError('Invalid lexical result')
            shared = self.policy.shared(query.text, self.passages[hit.passage_id].text)
            rows.append((hit, shared, hit.score > 0 and bool(shared)))
        return rows

    def search(self, query, k):
        return [hit.model_copy(update={'score': hit.score if allowed else 0.0,
                                      'retriever_id': self.retriever_id})
                for hit, _, allowed in self.decisions(query, k)]


class ContentGatedRRFRetriever(RRFRetriever):
    """Equal RRF with positive lexical score AND at least one shared content token."""
    def __init__(self, lexical, dense, constant=60, require_complete=True):
        self.content_policy = ContentTokenPolicy()
        super().__init__(_ContentMask(lexical, self.content_policy), dense,
                         constant=constant, gate_nonpositive_lexical=True,
                         require_complete=require_complete)

    def gate_diagnostics(self, query):
        if self._passages is None:
            raise ValueError('Index first')
        return [{'passage_id': hit.passage_id, 'original_rank': hit.rank,
                 'original_lexical_score': hit.score, 'shared_content_tokens': shared,
                 'lexical_contribution_allowed': allowed}
                for hit, shared, allowed in self.lexical.decisions(query, len(self._passages))]
