'use strict';
const form = document.querySelector('#search-form');
const question = document.querySelector('#question');
const language = document.querySelector('#language');
const submit = document.querySelector('#submit');
const results = document.querySelector('#results');
const status = document.querySelector('#status');
const count = document.querySelector('#result-count');
const error = document.querySelector('#input-error');
const answerPanel = document.querySelector('#answer-panel');
const mode = document.querySelector('#mode');
const crop = document.querySelector('#crop');
const collectionInfo = document.querySelector('#collection-info');
const coverageInfo = document.querySelector('#coverage-info');
const normalizeHinglish = document.querySelector('#normalize-hinglish');
const expandTerms = document.querySelector('#expand-terms');
const searchInterpretation = document.querySelector('#search-interpretation');
let busy = false;
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function panel(title, message, isError = false) {
  const node = el('div', undefined, `empty${isError ? ' error-panel' : ''}`);
  node.append(el('h3', title), el('p', message));
  results.replaceChildren(node);
}
function render(items, real = false) {
  results.replaceChildren();
  for (const item of items) {
    const card = el('article', undefined, 'evidence-card');
    const top = el('div', undefined, 'card-top');
    const heading = el('div');
    heading.append(el('span', real ? 'RETRIEVED EVIDENCE · LOCAL PAU' : 'DEMO / SYNTHETIC', 'tag'), el('h3', real ? `${item.metadata.crop || 'wheat'} · ${item.title}` : item.title));
    top.append(el('span', String(item.rank).padStart(2, '0'), 'rank'), heading);
    card.append(top, el('p', item.text, 'passage'), el('div', item.citation, 'source-line'));
    for (const context of (real ? item.metadata.source_qualifications || [] : [])) {
      const note = el('aside', undefined, 'source-qualification');
      note.append(el('strong', 'Required source qualification — read with this passage'),
        el('p', context.text),
        el('p', `${context.filename} · PDF page ${context.physical_page} · canonical interval ${context.evidence_span.start_char}–${context.evidence_span.end_char}`, 'source-line'));
      card.append(note);
    }
    if (real && item.metadata.conditions_and_limitations) {
      const limits = item.metadata.conditions_and_limitations;
      const context = el('aside', undefined, 'source-qualification');
      context.append(el('strong', 'Source conditions and limitations'));
      for (const value of [limits.conditions, limits.remaining_concerns]) {
        if (value) context.append(el('p', value));
      }
      card.append(context);
    }
    const details = el('details');
    const list = el('dl');
    for (const [label, value] of [['Document ID', item.document_id], ['Passage ID', item.passage_id], ['Section', item.section], ['Source type', real ? 'Restricted local PAU source · not a verified answer' : 'Invented UI fixture · not a real publication'], ...(real ? [['Canonical interval', `${item.evidence_span.start_char}–${item.evidence_span.end_char}`], ['Source SHA-256', item.evidence_span.content_sha256], ['Source URL', item.source_url || 'Not provided'], ['Printed page', item.metadata.printed_page_label || 'Not recorded'], ['Crop', item.metadata.crop || 'wheat'], ['Curation', item.metadata.curation || 'Inherited pilot source selection'], ['Human review', item.metadata.human_review || 'See original pilot record'], ['Source region ID', item.metadata.region_id || 'Not recorded'], ['Publication date', item.document_metadata.publication_date || 'Not recorded'], ['Applicability', (item.document_metadata.geographic_applicability || []).join('; ')], ['Source conditions', (item.metadata.conditions_and_limitations || {}).conditions || 'Review original context'], ['Remaining concerns', (item.metadata.conditions_and_limitations || {}).remaining_concerns || 'Review original context'], ['License status', item.document_metadata.license_status || 'Unknown']] : [])]) {
      list.append(el('dt', label), el('dd', value));
    }
    details.append(el('summary', 'Source & citation details'), list);
    card.append(details);
    results.append(card);
  }
}
question.addEventListener('input', () => { error.hidden = true; question.removeAttribute('aria-invalid'); });
question.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!busy) form.requestSubmit();
  }
});
const examples = {
  en: {soil: 'What are the fictional soil sample labels?', water: 'Show the water observation notes.', seed: 'What are the fictional seed batch names?'},
  hi: {soil: 'काल्पनिक मिट्टी के नमूनों के नाम क्या हैं?', water: 'पानी के अवलोकन वाले डेमो नोट दिखाएँ।', seed: 'काल्पनिक बीज के बैचों के नाम क्या हैं?'},
  hinglish: {soil: 'Fictional mitti ke samples ke naam kya hain?', water: 'Paani ke observation wale demo notes dikhao.', seed: 'Fictional beej batches ke naam kya hain?'}
};
language.addEventListener('change', () => { question.lang = language.value === 'hi' ? 'hi' : language.value === 'hinglish' ? 'hi-Latn' : 'en'; });
for (const button of document.querySelectorAll('[data-query]')) button.addEventListener('click', () => {
  mode.value = 'synthetic';
  question.value = examples[language.value][button.dataset.query];
  error.hidden = true; question.removeAttribute('aria-invalid'); question.focus();
});
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  const query = question.value.trim();
  if (!query || question.value.length > 1000) {
    error.textContent = 'Please enter a question between 1 and 1,000 characters.';
    error.hidden = false; question.setAttribute('aria-invalid', 'true'); question.focus(); return;
  }
  answerPanel.hidden = true; answerPanel.replaceChildren();
  if (searchInterpretation) { searchInterpretation.hidden = true; searchInterpretation.textContent = ''; }
  const real = mode.value === 'local_e5';
  mode.disabled = true; if (crop) crop.disabled = true;
  busy = true; submit.disabled = true; language.disabled = true; question.disabled = true;
  for (const b of document.querySelectorAll('[data-query]')) b.disabled = true;
  results.setAttribute('aria-busy', 'true'); count.textContent = '';
  status.textContent = real ? 'Searching local PAU evidence…' : 'Searching synthetic evidence…'; submit.textContent = 'Searching…';
  panel('Retrieving evidence', real ? 'Searching the selected local PAU collection with E5.' : 'Looking for matching synthetic topics.');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetch(real ? '/api/retrieve' : '/api/search', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query, language: language.value, ...(crop && real ? {crop:crop.value} : {}), ...(real && normalizeHinglish ? {normalize_hinglish:normalizeHinglish.checked} : {}), ...(real && expandTerms ? {expand_terms:expandTerms.checked} : {})}), signal: controller.signal});
    if (!response.ok) throw new Error('Request failed');
    const data = await response.json();
    if (data.mode !== (real ? 'local_e5' : 'synthetic') || !Array.isArray(data.results)) throw new Error('Invalid retrieval response');
    count.textContent = `${data.results.length} ${real ? 'retrieved' : 'demo'} passages`;
    status.textContent = real ? 'Related passages, not a confirmed answer. Search returns the closest matches even when this collection lacks the requested information. Check the topic and source conditions below.' : 'Synthetic matches only. These are invented examples.';
    if (real && Array.isArray(data.availability_notices) && data.availability_notices.length) {
      status.textContent = data.availability_notices.map(n => n.message).join(' ') + ' Results remain related source passages, not confirmed answers.';
    }
    if (searchInterpretation && real && data.query_transformation) {
      const t = data.query_transformation;
      searchInterpretation.hidden = false;
      searchInterpretation.textContent = t.applied
        ? `Search vocabulary assistance used: “${t.search_query}”. Original question unchanged. This is a limited glossary, not a verified translation. Disable both assistance options to search the original wording.`
        : 'Searched your original wording without glossary changes.';
    }
    if (real && data.results.length) {
      const hindi = language.value === 'hi' || /[\u0900-\u097f]/.test(query);
      answerPanel.hidden = false;
      answerPanel.lang = hindi ? 'hi' : 'en';
      answerPanel.append(el('span', hindi ? 'आपकी खोज का परिणाम' : 'YOUR SEARCH, WITH CONTEXT', 'tag'),
        el('h2', hindi ? 'स्रोत मिले हैं — पुष्टि किया हुआ उत्तर नहीं' : 'Source passages found'),
        el('p', hindi
          ? 'नीचे आपके प्रश्न से संबंधित मूल अंग्रेज़ी अंश हैं। ये पूरे प्रश्न का उत्तर देते हैं या नहीं, यह जाँचना ज़रूरी है। मानव-समीक्षित हिंदी सारांश अभी उपलब्ध नहीं हैं; अंग्रेज़ी पाठ को हिंदी उत्तर के रूप में प्रस्तुत नहीं किया गया है।'
          : 'Start with the passages below. Each keeps its original wording, page citation and conditions. A close match may cover only part of your question.'),
        el('p', hindi ? 'लागू क्षेत्र: पंजाब · स्रोत सत्र: रबी 2025–26। यह व्यक्तिगत कृषि सलाह नहीं है।' : 'Punjab · Rabi 2025–26. Source evidence, not personalized agricultural advice.', 'response-scope'));
      if (hindi) {
        count.textContent = `${data.results.length} स्रोत अंश`;
        status.textContent = 'मूल अंग्रेज़ी स्रोत और उनकी शर्तें नीचे देखें। स्रोत का मिलना पूर्ण उत्तर या कृषि संबंधी सत्यापन नहीं है.' + (data.availability_notices?.length ? ' '+data.availability_notices.map(n=>n.message).join(' ') : '');
      }
    }
    if (data.results.length) render(data.results, real);
    else panel('No passages found', 'No evidence was returned. This does not establish whether an answer exists elsewhere.');
  } catch (_) {
    status.textContent = 'Search could not be completed.';
    panel('Local retrieval is unavailable', 'Check the backend, or explicitly select the synthetic demo fallback. No alternative results were substituted.', true);
  } finally {
    clearTimeout(timer); mode.disabled = false; if (crop) crop.disabled = false; busy = false; submit.disabled = false; language.disabled = false; question.disabled = false;
    for (const b of document.querySelectorAll('[data-query]')) b.disabled = false;
    results.setAttribute('aria-busy', 'false'); submit.textContent = language.value === 'hi' ? 'स्रोत खोजें' : 'Find evidence';
  }
});

// Metadata only: no source text, query history or model downloads at startup.
if (collectionInfo && crop) {
  fetch('/api/status').then(response => {
    if (!response.ok) throw new Error('Unavailable');
    return response.json();
  }).then(data => {
    if (!data.live_available || !data.collection) {
      collectionInfo.textContent = 'Real evidence unavailable. Select the synthetic fallback explicitly.';
      return;
    }
    const c = data.collection;
    if (coverageInfo) {
      const updateCoverage = () => {
        coverageInfo.textContent = (c.coverage_notes || {})[crop.value] || 'Coverage is limited. The closest passages may not contain the requested information.';
      };
      crop.addEventListener('change', updateCoverage);
      updateCoverage();
    }
    collectionInfo.textContent = `${c.version} · ${c.passage_count} passages · ${Object.keys(c.crop_counts).length} crop groups. ${c.curation}`;
    const labels = {wheat:'Wheat / गेहूँ',barley:'Barley / जौ',maize:'Spring maize / मक्का',chickpea:'Chickpea / चना',mustard:'Rapeseed–mustard / सरसों'};
    for (const [key, n] of Object.entries(c.crop_counts)) {
      const option = el('option', `${labels[key] || key} (${n})`);
      option.value = key; crop.append(option);
    }
  }).catch(() => { collectionInfo.textContent = 'Collection status unavailable; check the local backend.'; });
}
