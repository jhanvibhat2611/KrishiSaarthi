"""Application-only, bounded vocabulary expansion; no answer or intent inference."""
import re

VERSION = 'application-search-terms-v2'
# Deliberately avoid ambiguous standalone words such as water, cut, feed or plant.
# No crop names, source IDs, scores, labels, quantities or recommended actions.
TERMS = (
    (r'\b(?:watered|waterings?|irrigate|irrigated|irrigating)\b', 'irrigation'),
    (r'\b(?:planting|planted|sowed)\b', 'sowing'),
    (r'\b(?:harvest|harvested)\b', 'harvesting'),
    (r'\b(?:row gap|row distance|rows? apart|line spacing|between (?:the )?rows|line to line distance)\b', 'row spacing'),
    (r'\b(?:plant to plant distance|between (?:the )?plants)\b', 'plant spacing'),
    (r'\bfertilis(?:er|ers|ation)\b', 'fertilizer'),
    (r'\b(?:seed quantity|quantity of seeds|amount of seed)\b', 'seed rate'),
)


def expand(transformation, language, enabled=True):
    """Append distinct terminology without replacing the original search wording."""
    base = transformation['search_query']
    terms=[]
    if enabled and language in ('en', 'hinglish'):
        for pattern, term in TERMS:
            if re.search(pattern, base, re.IGNORECASE) and not re.search(
                    r'\b'+re.escape(term)+r'\b', base, re.IGNORECASE):
                terms.append(term)
    search = base + ('\nRelated search terms: '+', '.join(terms) if terms else '')
    return {**transformation, 'search_query':search,
            'applied':search != transformation['original_query'],
            'terminology_version':VERSION, 'expanded_terms':terms}
