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
    heading.append(el('span', real ? 'RETRIEVED EVIDENCE · PAU PILOT' : 'DEMO / SYNTHETIC', 'tag'), el('h3', item.title));
    top.append(el('span', String(item.rank).padStart(2, '0'), 'rank'), heading);
    card.append(top, el('p', item.text, 'passage'), el('div', item.citation, 'source-line'));
    const details = el('details');
    const list = el('dl');
    for (const [label, value] of [['Document ID', item.document_id], ['Passage ID', item.passage_id], ['Section', item.section], ['Source type', real ? 'Restricted local PAU source · not a verified answer' : 'Invented UI fixture · not a real publication'], ...(real ? [['Canonical interval', `${item.evidence_span.start_char}–${item.evidence_span.end_char}`], ['Source SHA-256', item.evidence_span.content_sha256], ['Source URL', item.source_url || 'Not provided'], ['Printed page', item.metadata.printed_page_label || 'Not recorded'], ['Source region ID', item.metadata.region_id || 'Not recorded'], ['Publication date', item.document_metadata.publication_date || 'Not recorded'], ['Applicability', (item.document_metadata.geographic_applicability || []).join('; ')], ['Source conditions', (item.metadata.conditions_and_limitations || {}).conditions || 'Review original context'], ['Remaining concerns', (item.metadata.conditions_and_limitations || {}).remaining_concerns || 'Review original context'], ['License status', item.document_metadata.license_status || 'Unknown']] : [])]) {
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
  const real = mode.value === 'local_e5';
  mode.disabled = true;
  busy = true; submit.disabled = true; language.disabled = true; question.disabled = true;
  for (const b of document.querySelectorAll('[data-query]')) b.disabled = true;
  results.setAttribute('aria-busy', 'true'); count.textContent = '';
  status.textContent = real ? 'Searching local PAU evidence…' : 'Searching synthetic evidence…'; submit.textContent = 'Searching…';
  panel('Retrieving evidence', real ? 'Searching the frozen local PAU collection with E5.' : 'Looking for matching synthetic topics.');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetch(real ? '/api/retrieve' : '/api/search', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query, language: language.value}), signal: controller.signal});
    if (!response.ok) throw new Error('Request failed');
    const data = await response.json();
    if (data.mode !== (real ? 'local_e5' : 'synthetic') || !Array.isArray(data.results)) throw new Error('Invalid retrieval response');
    count.textContent = `${data.results.length} ${real ? 'retrieved' : 'demo'} passages`;
    status.textContent = real ? 'Retrieved evidence only. Relevance, completeness and current applicability require review.' : 'Synthetic matches only. These are invented examples.';
    if (data.results.length) render(data.results, real);
    else panel('No passages found', 'No evidence was returned. This does not establish whether an answer exists elsewhere.');
  } catch (_) {
    status.textContent = 'Search could not be completed.';
    panel('Local retrieval is unavailable', 'Check the backend, or explicitly select the synthetic demo fallback. No alternative results were substituted.', true);
  } finally {
    clearTimeout(timer); mode.disabled = false; busy = false; submit.disabled = false; language.disabled = false; question.disabled = false;
    for (const b of document.querySelectorAll('[data-query]')) b.disabled = false;
    results.setAttribute('aria-busy', 'false'); submit.textContent = 'Search Evidence ↗';
  }
});
