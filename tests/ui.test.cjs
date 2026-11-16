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
function setup(fetch) {
  const nodes = Object.fromEntries(['search-form','question','language','submit','results','status','result-count','input-error','answer-panel','mode'].map(x=>[x,new Element()]));
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
  assert.equal(endpoint,'/api/retrieve'); assert.equal(nodes['answer-panel'].hidden,true);
  const tree=JSON.stringify(nodes.results.children); assert.ok(tree.includes('RETRIEVED EVIDENCE')); assert.ok(tree.includes('10–20'));
});
