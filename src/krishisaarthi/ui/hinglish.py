"""Small, transparent application search glossary, not a general translator.

Exact word/phrase boundaries only. Unknown words, numbers, varieties and units
are retained. No evidence, annotation, expected answer or query ID is consulted.
"""
import re

VERSION = 'application-hinglish-glossary-v2'
# Ambiguous standalone words (e.g. "jau" = barley / "shall I go") are omitted.
# No stemming, fuzzy replacement, inferred quantities or inferred crop filters.
GLOSSARY = {
    'gehu': 'wheat', 'gehoon': 'wheat', 'gehun': 'wheat', 'gehuun': 'wheat',
    'gehoo': 'wheat', 'gehूं': 'wheat', 'gehun ki': 'wheat',
    'makka': 'maize', 'makai': 'maize', 'makkai': 'maize',
    'chana': 'chickpea', 'chane': 'chickpea', 'sarson': 'mustard',
    'sarso': 'mustard', 'sarson ki': 'mustard',
    'line se line': 'row to row', 'paudhe se paudhe': 'plant to plant',
    'doori': 'spacing', 'duri': 'spacing', 'faasla': 'spacing', 'fasla': 'spacing',
    'barish': 'rain', 'baarish': 'rain', 'pehla': 'first',
    'konse': 'which', 'kaatne': 'harvesting',
    'dobara bone': 're-sowing', 'barish ke bharose': 'rainfed',
    'baarish ke bharose': 'rainfed',
    'kab': 'when', 'kaise': 'how', 'kitna': 'how much', 'kitni': 'how much',
    'kitne': 'how many', 'kaunsa': 'which', 'kaunsi': 'which',
    'kaunse': 'which', 'konsa': 'which', 'konsi': 'which', 'kya': 'what',
    'buvai': 'sowing', 'buwai': 'sowing', 'buai': 'sowing', 'boyai': 'sowing',
    'bona': 'sow', 'bone': 'sowing', 'boein': 'sow', 'boye': 'sow',
    'ugana': 'grow', 'ugane': 'growing', 'ugaye': 'grow',
    'katai': 'harvesting', 'katayi': 'harvesting', 'kaatna': 'harvest',
    'sinchai': 'irrigation', 'sichai': 'irrigation', 'paani': 'water', 'pani': 'water',
    'mitti': 'soil', 'beej': 'seed', 'khad': 'fertilizer', 'khaad': 'fertilizer',
    'fasal': 'crop', 'khet': 'field', 'patte': 'leaves', 'patti': 'leaf',
    'peele': 'yellow', 'pili': 'yellow', 'peeli': 'yellow', 'bimari': 'disease',
    'bimaari': 'disease', 'rog': 'disease', 'kami': 'deficiency',
    'bhari': 'heavy', 'halki': 'light', 'sukhi': 'dry', 'geeli': 'wet',
    'der se': 'late', 'jaldi': 'early', 'pehle': 'before', 'baad': 'after',
    'se kam': 'less than', 'se zyada': 'greater than', 'se jyada': 'greater than',
    'nahi': 'not', 'nahin': 'not', 'naheen': 'not', 'mat': 'do not',
    'aur': 'and', 'ya': 'or', 'agar': 'if', 'sirf': 'only', 'bina': 'without',
    'sabse achha': 'best', 'sabse acha': 'best', 'sahi': 'suitable',
    'samay': 'time', 'mausam': 'season', 'mahina': 'month', 'din': 'days',
    'chahiye': 'should', 'chahie': 'should', 'karna': 'do', 'kare': 'do',
    'dena': 'give', 'dale': 'apply', 'daalna': 'apply',
    'mein': 'in', 'me': 'in', 'ke liye': 'for', 'ka': 'of', 'ki': 'of', 'ke': 'of',
    'hai': 'is', 'hain': 'are', 'hota': 'occurs', 'hoti': 'occurs',
}
PATTERN = re.compile(r'(?<!\w)(?:' + '|'.join(
    re.escape(k).replace(r'\ ', r'\s+') for k in sorted(GLOSSARY, key=len, reverse=True)
) + r')(?!\w)', re.IGNORECASE)


def normalize(question, language, enabled=True, crop=None):
    """Return original and exact search text, including auditable replacements."""
    replacements = []
    def replace(match):
        original = match.group()
        replacement = GLOSSARY[re.sub(r'\s+', ' ', original.casefold())]
        replacements.append({'original':original, 'replacement':replacement})
        return replacement
    search = PATTERN.sub(replace, question) if enabled and language == 'hinglish' else question
    # Disambiguate only using an explicit barley filter; never guess from 'jau'.
    if enabled and language == 'hinglish' and crop == 'barley':
        def barley(match):
            replacements.append({'original':match.group(), 'replacement':'barley'})
            return 'barley'
        search = re.sub(r'(?<!\w)jau(?!\w)', barley, search, flags=re.I)
    return {'version':VERSION, 'original_query':question, 'search_query':search,
            'applied':search != question, 'replacements':replacements,
            'notice':'Glossary-assisted search, not a verified translation. Unknown words are retained.'}
