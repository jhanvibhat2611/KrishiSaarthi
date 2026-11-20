"""Capability notices for a static PAU collection, not answerability labels.

Rules inspect original wording only and never suppress/reorder search results.
No matching notice must never be interpreted as evidence sufficiency.
"""
import re

VERSION = 'static-pau-capabilities-v2'
LIVE = re.compile(r'\b(today|tomorrow|current|currently|latest|live|now|aaj|aajkal|abhi|kal|taaza|taza)\b|आज|कल|वर्तमान|ताज़ा|ताजा|अभी', re.I)
PRICE = re.compile(r'\b(prices?|mandi|bhaav|bhav|daam|keemat|kimat)\b|मंडी|कीमत|भाव|दाम', re.I)
WEATHER = re.compile(r'\b(weather|forecast|rain|raining|mausam|barish|baarish)\b|मौसम|बारिश|पूर्वानुमान', re.I)
MESSAGES = {
    'live_prices': 'Current market prices unavailable: this static PAU collection has no live mandi-price feed. Cultivation passages below cannot establish a current price.',
    'live_weather': 'Live weather and forecasts unavailable: this static PAU collection has no weather feed. Source climate descriptions cannot establish local weather now or a forecast.',
}


def notices(question, crop="all"):
    findings=[]
    if LIVE.search(question) and PRICE.search(question):
        findings.append({'code':'live_prices','message':MESSAGES['live_prices'],'policy':VERSION})
    if WEATHER.search(question) and (LIVE.search(question) or re.search(r'\bforecast\b|पूर्वानुमान',question,re.I)):
        findings.append({'code':'live_weather','message':MESSAGES['live_weather'],'policy':VERSION})
    # Bounded literal mentions: notices ask for clarification, never change filters.
    aliases = {
        'wheat': r'\bwheat\b|\bgehu[n]?\b|गेहूँ|गेहूं',
        'barley': r'\bbarley\b|जौ',
        'maize': r'\b(?:maize|makka|makai)\b|मक्का',
        'chickpea': r'\b(?:chickpeas?|gram|chana|chane)\b|चना',
        'mustard': r'\b(?:mustard|rapeseed|sarson|sarso|toria)\b|सरसों|तोरिया',
    }
    mentioned = [name for name, pattern in aliases.items() if re.search(pattern, question, re.I)]
    def add(code, message):
        findings.append({'code':code, 'message':message, 'policy':VERSION})
    if crop != 'all' and any(name != crop for name in mentioned):
        add('crop_filter_check', 'Check the crop filter: your question also mentions a different crop. Results remain restricted to your selected crop; comparisons or rotation questions may need separate searches.')
    if re.search(r'\b(?:tomatoes|tomato|rice|paddy|cotton|potato|potatoes)\b|टमाटर|धान|चावल|कपास|आलू', question, re.I):
        add('outside_crop_scope', 'This collection covers wheat, barley, spring maize, chickpea and rapeseed–mustard only. Other crop mentions may describe rotations; they do not establish guidance for that crop.')
    if re.search(r'\b(?:Rajasthan|Tamil Nadu|Bengaluru|Bangalore|Assam|Meghalaya)\b|राजस्थान|तमिलनाडु|बेंगलुरु|असम|मेघालय', question, re.I):
        add('outside_region_scope', 'These sources are Punjab-specific. Retrieved passages do not establish applicability to the other region mentioned.')
    if re.search(r'\b(?:fungicides?|pesticides?|diagnos\w*|treatments?)\b|फफूंदनाशक|कीटनाशक|निदान|इलाज', question, re.I):
        add('treatment_evidence_check', 'Disease descriptions alone cannot establish a diagnosis or treatment. Check for explicit disease-specific instructions, all conditions and source date; this app does not verify current treatment suitability.')
    return findings
