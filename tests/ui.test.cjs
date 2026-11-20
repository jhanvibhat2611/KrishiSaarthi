// Dependency-free controller tests. Browser layout is verified separately.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
class Element {
  constructor() { this.value=''; this.children=[]; this.events={}; this.attrs={}; this.hidden=true; this.disabled=false; this.dataset={}; }
  addEventListener(type, fn) { this.events[type]=fn; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=nodes; }
  setAttribute(k,v) { this.attrs[k]=v; }
  removeAttribute(k) { delete this.attrs[k]; }
  focus() { this.focused=true; }
}
function setup(fetch, withCollection = false) {
  const nodes = Object.fromEntries(['search-form','question','language','submit','results','status','result-count','input-error','answer-panel','mode'].map(x=>[x,new Element()]));
  if(withCollection){nodes.crop=new Element();nodes.crop.value='all';nodes['collection-info']=new Element();nodes['normalize-hinglish']=new Element();nodes['normalize-hinglish'].checked=true;nodes['expand-terms']=new Element();nodes['expand-terms'].checked=true;nodes['search-interpretation']=new Element();}
  nodes.language.value='en'; nodes.mode.value='synthetic'; const examples=['soil','water','seed'].map(x=>{const e=new Element();e.dataset.query=x;return e;});
  nodes['search-form'].requestSubmit=()=>nodes['search-form'].events.submit({preventDefault(){}});
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../src/krishisaarthi/ui/static/app.js'),'utf8'),{document:{querySelector:s=>nodes[s.slice(1)],querySelectorAll:()=>examples,createElement:()=>new Element()},fetch,setTimeout,clearTimeout,AbortController});
  return {nodes,examples,send:()=>nodes['search-form'].requestSubmit()};
}
const result={rank:1,title:'Synthetic title',text:'SYNTHETIC invented content',document_id:'demo-1',passage_id:'synthetic-1',section:'Entry 1',citation:'DEMO · demo-1'};
const response=items=>({ok:true,json:async()=>({mode:'synthetic',results:items,qa:{mode:'synthetic_test_double',status:'UNANSWERABLE',reason:'Finite demo only',citations:[],language:'en',query_type:'outside_fixture_scope'}})});
test('invalid input does not submit',async()=>{let calls=0;const {nodes,send}=setup(()=>calls++);await send();assert.equal(calls,0);assert.equal(nodes['input-error'].hidden,false);assert.equal(nodes.question.attrs['aria-invalid'],'true');});
test('three language choices and examples retain language payload',async()=>{for(const language of ['en','hi','hinglish']){let payload;const {nodes,examples,send}=setup(async(_,opts)=>{payload=JSON.parse(opts.body);return response([result]);});nodes.language.value=language;nodes.language.events.change();examples[0].events.click();await send();assert.equal(payload.language,language);assert.ok(payload.query);assert.equal(nodes.results.children.length,1);assert.equal(nodes.results.attrs['aria-busy'],'false');}});
test('loading prevents duplicate submission then renders citations',async()=>{let release,calls=0;const {nodes,send}=setup(()=>{calls++;return new Promise(r=>release=r);});nodes.question.value='soil';const first=send();assert.equal(nodes.submit.disabled,true);assert.equal(nodes.results.attrs['aria-busy'],'true');await send();assert.equal(calls,1);release(response([result]));await first;assert.equal(nodes.submit.disabled,false);const tree=JSON.stringify(nodes.results.children);assert.ok(tree.includes('DEMO · demo-1'));assert.ok(tree.includes('synthetic-1'));assert.ok(tree.includes('DEMO / SYNTHETIC'));});
test('empty and backend-error states',async()=>{for(const failure of [false,true]){const {nodes,send}=setup(async()=>{if(failure)throw Error('offline');return response([]);});nodes.question.value='x';await send();const tree=JSON.stringify(nodes.results.children);assert.ok(tree.includes(failure?'Local retrieval is unavailable':'No passages found'));assert.equal(nodes.submit.disabled,false);}});
test('Enter submits; Shift+Enter and composition do not',()=>{const {nodes}=setup();let calls=0;nodes['search-form'].requestSubmit=()=>calls++;const key=nodes.question.events.keydown;key({key:'Enter',shiftKey:true});key({key:'Enter',isComposing:true});assert.equal(calls,0);key({key:'Enter',preventDefault(){}});assert.equal(calls,1);});
test('unexpected live mode rejected',async()=>{const {nodes,send}=setup(async()=>({ok:true,json:async()=>({mode:'live',results:[result]})}));nodes.question.value='soil';await send();assert.ok(JSON.stringify(nodes.results.children).includes('Local retrieval is unavailable'));});

test('real evidence route and provenance; no answer generation', async()=>{
  let endpoint;
  const {nodes,send}=setup(async(url)=>{endpoint=url;return {ok:true,json:async()=>({mode:'local_e5',results:[{...result,evidence_span:{start_char:10,end_char:20,content_sha256:'synthetic-hash'},document_metadata:{region:'synthetic region'},metadata:{}}]})};});
  nodes.mode.value='local_e5'; nodes.question.value='synthetic'; await send();
  assert.equal(endpoint,'/api/retrieve'); assert.equal(nodes['answer-panel'].hidden,false); assert.ok(JSON.stringify(nodes['answer-panel']).includes('Source passages found'));
  const tree=JSON.stringify(nodes.results.children); assert.ok(tree.includes('RETRIEVED EVIDENCE')); assert.ok(tree.includes('10–20'));
});

test('live collection metadata and crop filter remain explicit',async()=>{
  let payload;
  const {nodes,send}=setup(async(url,opts)=>{
    if(url==='/api/status')return {ok:true,json:async()=>({live_available:true,collection:{version:'SYNTHETIC collection',passage_count:12,crop_counts:{wheat:5,barley:7},curation:'AI-assisted; not human-verified'}})};
    payload=JSON.parse(opts.body);
    return {ok:true,json:async()=>({mode:'local_e5',results:[]})};
  },true);
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(nodes['collection-info'].textContent.includes('12 passages'));
  assert.ok(nodes['collection-info'].textContent.includes('not human-verified'));
  assert.equal(nodes.crop.children.length,2);
  nodes.mode.value='local_e5';nodes.crop.value='barley';nodes.question.value='SYNTHETIC';
  await send();assert.equal(payload.crop,'barley');assert.equal(nodes.crop.disabled,false);
});

test('required source qualification is visible outside collapsed citation details', async()=>{
  const {nodes,send}=setup(async()=>({ok:true,json:async()=>({mode:'local_e5',results:[{...result,evidence_span:{start_char:0,end_char:10,content_sha256:'synthetic'},document_metadata:{},metadata:{source_qualifications:[{text:'SYNTHETIC qualification only',filename:'fixture.txt',physical_page:2,evidence_span:{start_char:100,end_char:140}}]}}]})}));
  nodes.mode.value='local_e5';nodes.question.value='synthetic';await send();
  const card=nodes.results.children[0];
  assert.ok(card.children.some(x=>x.className==='source-qualification'));
  assert.ok(JSON.stringify(card).includes('100–140'));
  assert.ok(nodes.status.textContent.includes('not a confirmed answer'));
});


test('Hinglish assistance is visible and can be disabled without editing the question',async()=>{
 let payload;
 const {nodes,send}=setup(async(url,opts)=>{
  if(url==='/api/status')return {ok:true,json:async()=>({live_available:false})};
  payload=JSON.parse(opts.body);
  return {ok:true,json:async()=>({mode:'local_e5',results:[],query_transformation:{applied:payload.normalize_hinglish,search_query:'SYNTHETIC normalized query'}})};
 },true);
 nodes.mode.value='local_e5';nodes.language.value='hinglish';nodes.question.value='SYNTHETIC original';
 await send();assert.equal(payload.normalize_hinglish,true);
 assert.ok(nodes['search-interpretation'].textContent.includes('SYNTHETIC normalized query'));
 assert.equal(nodes.question.value,'SYNTHETIC original');
 nodes['normalize-hinglish'].checked=false;await send();
 assert.equal(payload.normalize_hinglish,false);
 assert.ok(nodes['search-interpretation'].textContent.includes('original wording'));
});


test('everyday terminology expansion is disclosed and independently switchable',async()=>{
 let payload;
 const {nodes,send}=setup(async(url,opts)=>{
  if(url==='/api/status')return {ok:true,json:async()=>({live_available:false})};
  payload=JSON.parse(opts.body);
  return {ok:true,json:async()=>({mode:'local_e5',results:[],query_transformation:{applied:payload.expand_terms,search_query:'SYNTHETIC watered. Related search terms: irrigation'}})};
 },true);
 nodes.mode.value='local_e5';nodes.question.value='SYNTHETIC watered';
 await send();assert.equal(payload.expand_terms,true);
 assert.ok(nodes['search-interpretation'].textContent.includes('irrigation'));
 assert.equal(nodes.question.value,'SYNTHETIC watered');
 nodes['expand-terms'].checked=false;await send();assert.equal(payload.expand_terms,false);
 assert.ok(nodes['search-interpretation'].textContent.includes('original wording'));
});

test('live-data notice is specific without suppressing retrieved evidence',async()=>{
 const {nodes,send}=setup(async()=>({ok:true,json:async()=>({mode:'local_e5',availability_notices:[{message:'Current market prices unavailable: SYNTHETIC capability notice.'}],results:[{...result,metadata:{},document_metadata:{},evidence_span:{start_char:0,end_char:10,content_sha256:'fixture'}}]})}));
 nodes.mode.value='local_e5';nodes.question.value='SYNTHETIC current price';await send();
 assert.ok(nodes.status.textContent.includes('Current market prices unavailable'));
 assert.equal(nodes.results.children.length,1);
 assert.ok(JSON.stringify(nodes.results.children).includes('synthetic-1'));
});

test('source limitations stay visible and scope notices do not change crop filter',async()=>{
 const {nodes,send}=setup(async()=>({ok:true,json:async()=>({mode:'local_e5',availability_notices:[{message:'Check the crop filter: SYNTHETIC.'}],results:[{...result,metadata:{conditions_and_limitations:{conditions:'SYNTHETIC dry setup only',remaining_concerns:'SYNTHETIC surrounding context required'}},document_metadata:{},evidence_span:{start_char:0,end_char:10,content_sha256:'fixture'}}]})}),true);
 nodes.mode.value='local_e5';nodes.crop.value='barley';nodes.question.value='SYNTHETIC';await send();
 const aside=nodes.results.children[0].children.find(x=>x.className==='source-qualification');
 assert.ok(JSON.stringify(aside).includes('dry setup only'));
 assert.ok(JSON.stringify(aside).includes('surrounding context required'));
 assert.equal(nodes.crop.value,'barley');
 assert.ok(nodes.status.textContent.includes('Check the crop filter'));
 assert.ok(!nodes.status.textContent.includes('live-data request'));
});


test('Hindi response guidance does not fabricate a translated source answer',async()=>{
 const {nodes,send}=setup(async()=>({ok:true,json:async()=>({mode:'local_e5',results:[{...result,metadata:{},document_metadata:{},evidence_span:{start_char:0,end_char:10,content_sha256:'fixture'}}]})}));
 nodes.mode.value='local_e5';nodes.language.value='hi';nodes.question.value='काल्पनिक प्रश्न';await send();
 assert.equal(nodes['answer-panel'].lang,'hi');
 assert.ok(JSON.stringify(nodes['answer-panel']).includes('मानव-समीक्षित हिंदी सारांश अभी उपलब्ध नहीं'));
 assert.ok(JSON.stringify(nodes.results).includes(result.text));
 assert.equal(nodes.submit.textContent,'स्रोत खोजें');
});
