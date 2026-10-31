"""One pinned dense model per process. Validates all frozen inputs before inference."""
from krishisaarthi.validation import require
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,hashlib,json,platform,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from krishisaarthi.config import fingerprint,load_config
from krishisaarthi.schemas import Document,Query,GoldLabel,EvidenceSpan
from krishisaarthi.ingestion.storage import read_jsonl
from krishisaarthi.evaluation.labels import validate_labels
from krishisaarthi.evaluation.metrics import fully_covered
from krishisaarthi.evaluation.runner import evaluate
from krishisaarthi.retrieval.dense import HFEncoder,DeviceFallbackRequired
from krishisaarthi.retrieval.e5 import E5Retriever
from krishisaarthi.retrieval.bge_m3 import BGEM3Retriever

ROOT=Path(__file__).resolve().parents[1]
CORPUS_FP='a919560be507b862be119a236fd1d2501f8d7c32d8fe6ea4951114b205f8c88a'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_ledger(directory):
    for name,h in json.loads((directory/'integrity.json').read_text()).items():
        p=(directory/name).resolve()
        if not p.is_relative_to(directory.resolve()) or sha(p)!=h:raise ValueError('Input integrity mismatch: '+name)

def run(args):
    start=time.perf_counter();c=ROOT/'data/pilot-v0.1';a=ROOT/'data/annotation-final-signoff-v1';b=ROOT/'data/experiments/bm25-pilot-v0.1-run-001'
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'data') or out.exists():raise ValueError('Use a fresh Git-ignored data output directory')
    verify_ledger(c);verify_ledger(b)
    protected={str(p):sha(p) for d in (c,a,b) for p in d.rglob('*') if p.is_file()}
    sign=json.loads((a/'reviewer-signoff.json').read_text());baseline=json.loads((b/'manifest.json').read_text())
    require(sign['status']=='signed_off' and sign['final_decision']=='accept', 'run_dense_pilot.py:36: integrity validation failed')
    require(sign['corpus_fingerprint']==CORPUS_FP==baseline['corpus_fingerprint'], 'run_dense_pilot.py:37: integrity validation failed')
    require(sha(a/'annotation-candidate.json')==sign['annotation_sha256']==baseline['annotation_sha256'], 'run_dense_pilot.py:38: integrity validation failed')
    require(sha(a/'reviewer-signoff.json')==baseline['signoff_sha256'], 'run_dense_pilot.py:39: integrity validation failed')
    docs=read_jsonl(c/'documents.jsonl',Document)
    from krishisaarthi.schemas import Passage
    passages=read_jsonl(c/'passages.jsonl',Passage);queries=read_jsonl(b/'queries.jsonl',Query);gold=read_jsonl(b/'gold.jsonl',GoldLabel)
    metadata=json.loads((b/'query-evidence-metadata.json').read_text())
    require(len(passages)==15 and len(queries)==48 and len(gold)==48 and sum(l.answerable for l in gold)==36, 'run_dense_pilot.py:44: integrity validation failed')
    require(len({m['group_id'] for m in metadata})==16, 'run_dense_pilot.py:45: integrity validation failed')
    require(not validate_labels(queries,gold,docs,passages)[2], 'run_dense_pilot.py:46: integrity validation failed')
    for m in metadata:
        if m['group_id'][-2:] in ['08','10']:require(m['evidence_requirement']=='ALL' and len(m['evidence'])==2, 'run_dense_pilot.py:48: integrity validation failed')
    eval_fp=fingerprint({'queries':[q.model_dump() for q in queries],'gold':[l.model_dump() for l in gold],'metadata':metadata})
    require(eval_fp==baseline['evaluation_set_fingerprint'], 'run_dense_pilot.py:50: integrity validation failed')
    spec=json.loads((ROOT/'configs/dense-models.json').read_text())[args.model]
    directory=ROOT/'data/models'/args.model/spec['revision']
    downloaded=json.loads((directory/'download-manifest.json').read_text());require(downloaded['model']==spec, 'run_dense_pilot.py:53: integrity validation failed')
    for n,h in downloaded['files_sha256'].items():require(sha(directory/n)==h, 'run_dense_pilot.py:54: integrity validation failed')
    spec={**spec,'checkpoint_files_fingerprint':fingerprint(downloaded['files_sha256'])}
    print('Preflight passed: frozen corpus, signed labels, BM25 results and pinned checkpoint hashes.',flush=True)
    t=time.perf_counter();encoder=HFEncoder(directory,spec,device=args.device,batch_size=args.batch_size);load_seconds=time.perf_counter()-t
    retriever=(E5Retriever if args.model=='e5' else BGEM3Retriever)(encoder,spec,CORPUS_FP,ROOT/'data/embeddings')
    retriever.preflight(passages,queries)
    print('All 63 inputs within token limits; device='+encoder.device,flush=True)
    retriever.prepare_queries(queries)
    cfg=load_config(b/'config.toml');require(cfg.evaluation.ks==[1,3,5], 'run_dense_pilot.py:62: integrity validation failed')
    t=time.perf_counter();report=evaluate(retriever,queries,gold,docs,passages,cfg.evaluation);evaluation_seconds=time.perf_counter()-t
    require(report['diagnostics']['available'] is False, 'run_dense_pilot.py:64: integrity validation failed')
    byid={m['query_id']:m for m in metadata};details=[]
    for q,l in zip(queries,gold):
        hits=list(retriever.search(q,15));refs=byid[q.query_id]['evidence']
        def covered(rows):return [r['evidence_id'] for r in refs if fully_covered(EvidenceSpan.model_validate(r['span']),[h.evidence_span for h in rows])]
        missing={str(k):[r['evidence_id'] for r in refs if r['evidence_id'] not in covered(hits[:k])] for k in [1,3,5]}
        details.append({'query_id':q.query_id,'group_id':byid[q.query_id]['group_id'],'language':q.language,'question':q.text,'answerable':l.answerable,
            'required_evidence_ids':[r['evidence_id'] for r in refs],'missing_evidence_ids':missing,
            'all_required_covered':{str(k):not missing[str(k)] if l.answerable else None for k in [1,3,5]},
            'full_ranking':[{**h.model_dump(),'covered_gold_evidence_ids':covered([h])} for h in hits]})
    require(all(sha(Path(p))==h for p,h in protected.items()), 'run_dense_pilot.py:74: integrity validation failed')
    out.mkdir(parents=True)
    def save(n,v):(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    save('evaluation.json',report);save('failure-analysis.json',details);save('token-lengths.json',retriever.token_lengths)
    (out/'config.toml').write_bytes((b/'config.toml').read_bytes())
    (out/'dependencies.lock').write_bytes((ROOT/'requirements-dense.lock').read_bytes())
    packages=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
    fallback_path=out.parent/(out.name+'-fallback.json')
    save('manifest.json',{'experiment':args.model+'-dense-only','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'corpus_fingerprint':CORPUS_FP,'evaluation_set_fingerprint':eval_fp,'query_fingerprint':report['query_fingerprint'],
        'label_fingerprint':report['label_fingerprint'],'annotation_sha256':sign['annotation_sha256'],'signoff_sha256':baseline['signoff_sha256'],
        'config':cfg.model_dump(),'model':spec,'model_fingerprint':retriever.model_fingerprint,'inference':encoder.identity,
        'checkpoint_files_sha256':downloaded['files_sha256'],'retriever_id':retriever.retriever_id,
        'cache_key':retriever.cache_key,'passage_cache_hit':retriever.cache_hit,
        'timings':{**retriever.timings,'model_loading_seconds':load_seconds,'evaluation_wall_seconds':evaluation_seconds,'total_wall_seconds':time.perf_counter()-start},
        'timing_notes':'Encoding includes model inference and transfers; ranking is cosine+sorting+result construction. Total includes input/checkpoint hashing. All timings single-run observations, not benchmarks.',
        'fallback':json.loads(fallback_path.read_text()) if fallback_path.exists() else None,
        'python':platform.python_version(),'platform':platform.platform(),'packages_freeze':packages,
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').rglob('*.py')},
        'script_sha256':sha(Path(__file__)),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'protected_inputs_unchanged':True,'score_policy':'Normalized float32 dense cosine; descending score then passage ID. No lexical/no-positive BM25 diagnostic semantics.'})
    save('integrity.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps({'model':args.model,'device':encoder.device,'metrics':report['per_language'],'timings':retriever.timings},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['e5','bge-m3'],required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['auto','mps','cpu'],default='auto');p.add_argument('--batch-size',type=int,default=2);args=p.parse_args()
    try:run(args)
    except DeviceFallbackRequired as exc:
        if args.device=='cpu':raise
        out=(ROOT/args.output).resolve()
        if not out.is_relative_to(ROOT/'data'):raise
        out.parent.mkdir(parents=True,exist_ok=True)
        (out.parent/(out.name+'-fallback.json')).write_text(json.dumps({'requested_device':args.device,'restart_device':'cpu','reason':str(exc),'policy':'Entire process restarted; no mixed-device embeddings.'},indent=2)+'\n')
        print('MPS inference failed; restarting whole run on CPU: '+str(exc),flush=True)
        os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),'--model',args.model,'--output',args.output,'--device','cpu','--batch-size',str(args.batch_size)])
