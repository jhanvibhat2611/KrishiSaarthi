"""Application-only source-context representation; research E5 remains unchanged."""
from .e5 import E5Retriever

REPRESENTATION = 'source-crop-topic-and-exact-text-v1'
CROP_LABELS = {'wheat':'Wheat','barley':'Barley','maize':'Spring maize',
               'chickpea':'Chickpea (gram)','mustard':'Rapeseed and mustard'}


class ApplicationE5(E5Retriever):
    def __init__(self, encoder, spec, corpus_fingerprint, cache_dir=None):
        super().__init__(encoder, {**spec, 'application_representation':REPRESENTATION}, corpus_fingerprint, cache_dir)
        self._source_context = {}

    def bind_context(self, passages):
        self._source_context = {}
        for p in passages:
            crop = p.domain_metadata.get('crop','wheat')
            if crop not in CROP_LABELS:
                raise ValueError('Unknown crop context')
            topic = p.domain_metadata.get('topic','Source evidence')
            context = f'Crop: {CROP_LABELS[crop]}. Source section: {topic}. Punjab, India. PAU Rabi 2025–26.\n'
            if p.text in self._source_context and self._source_context[p.text] != context:
                raise ValueError('Ambiguous duplicate source text context')
            self._source_context[p.text] = context

    def formatted(self, texts, kind):
        if kind == 'passage':
            # All metadata participates in the passage/cache fingerprint. Prefix,
            # special tokens and this context are included in token-limit checks.
            return [self.spec['passage_prefix'] + self._source_context[t] + t for t in texts]
        return super().formatted(texts,kind)

    def index(self, passages):
        self.bind_context(passages)
        return super().index(passages)
