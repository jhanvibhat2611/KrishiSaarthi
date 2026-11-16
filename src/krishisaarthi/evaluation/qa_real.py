"""Single approved local experiment. No retries, downloads, index creation or scoring."""
import json
import os
import re
import subprocess
import time
import resource
from pathlib import Path
from .qa_runtime import (proposal,check_local,sha,private_directory,PrivateHandoff,HandoffCase,
                         validate_handoff,DOCUMENTS_SHA,QUERIES_SHA,PASSAGES,CORPUS)
from .qa_adapters import require,E5EvidenceAdapter,QwenDecoder,PairedComparison,rag_prompt
from .qa_comparison_fake import Question,build_direct_prompt,fingerprint
from .qa_journal import durable_json,AttemptJournal
from ..schemas import Document,Passage,Query
from ..ingestion.storage import read_jsonl
from ..rag.models import EvidencePack

APPROVAL_SHA='f26d204fcd4e12012547f0613695f105abcaf8cfb3aa1c7ec6ff66ed987614be'
MARKER='qa-c3-one-authorized-run.json'


class Halt(ValueError):
    def __init__(self,code): self.code=code;super().__init__(code)


def inputs(root,p):
    c=root/'data/pilot-v0.1'
    require(sha(c/'documents.jsonl')==DOCUMENTS_SHA,'source_validation_failure')
    docs=read_jsonl(c/'documents.jsonl',Document);passages=read_jsonl(c/'passages.jsonl',Passage)
    require(fingerprint([x.model_dump() for x in passages])==PASSAGES,'source_validation_failure')
    qpath=root/'data/experiments/bm25-pilot-v0.1-run-001/queries.jsonl'
    require(sha(qpath)==QUERIES_SHA,'query_validation_failure')
    qmap={q.query_id:q for q in read_jsonl(qpath,Query)}
    questions={c.case_id:Question(text=qmap[c.case_id].text,language=c.language) for c in p.cases}
    return docs,passages,questions


def pack(case,h,p):
    return EvidencePack(question=Query(query_id=case.case_id,**case.question.model_dump()),passages=case.passages,
        retriever_id=p.index_identifier,document_metadata={d.document_id:d.domain_metadata for d in h.documents})


def resources(reserve_bytes=3*1024**3):
    import mlx.core as mx
    if not mx.metal.is_available():raise Halt('metal_unavailable')
    info=mx.metal.device_info()
    working=info.get('max_recommended_working_set_size',info.get('maxRecommendedWorkingSetSize',0))
    total=int(subprocess.check_output(['/usr/sbin/sysctl','-n','hw.memsize'],text=True))
    vm=subprocess.check_output(['/usr/bin/vm_stat'],text=True)
    page=int(re.search(r'page size of (\d+) bytes',vm).group(1))
    def pages(label):
        m=re.search(re.escape(label)+r':\s+(\d+)',vm)
        return int(m.group(1)) if m else 0
    available=page*sum(pages(k) for k in ('Pages free','Pages inactive','Pages speculative','Pages purgeable'))
    # Fixed engineering guards, not model-quality tuning; retain existing 6 GiB MLX limit.
    if total<12*1024**3 or working<6*1024**3 or available<reserve_bytes:raise Halt('insufficient_memory_headroom')
    return {'physical_memory_bytes':total,'recommended_working_set_bytes':working,
            'estimated_reclaimable_memory_bytes':available,'mlx_limit_bytes':6*1024**3,
            'rss_limit_bytes':8*1024**3,'required_reclaimable_bytes':reserve_bytes,'note':'vm_stat reclaimable estimate, not guaranteed allocatable memory'}


def producer(root,out):
    p=proposal(root);docs,passages,questions=inputs(root,p)
    from ..ui.local import LocalRetrieval
    started=time.perf_counter();service=LocalRetrieval(root);load=time.perf_counter()-started
    adapter=E5EvidenceAdapter(service,p);cases=[];timings=[]
    journal=AttemptJournal(out/'retrieval-attempts',[(c.case_id,'e5') for c in p.cases])
    for c in p.cases:
        journal.before(c.case_id,'e5');start=time.perf_counter()
        try:
            evidence=adapter.retrieve(questions[c.case_id])
            elapsed=time.perf_counter()-start
            cases.append(HandoffCase(case_id=c.case_id,question=questions[c.case_id],passages=evidence.passages))
            row={'case_id':c.case_id,'arm':'e5','status':'completed','retrieval_seconds':elapsed,
                 'passage_ids':[x.passage_id for x in evidence.passages]}
            journal.after(row);timings.append(row)
        except BaseException:
            journal.after({'case_id':c.case_id,'arm':'e5','status':'error','failure_category':'retrieval_or_source_failure',
                           'retrieval_seconds':time.perf_counter()-start})
            raise Halt('retrieval_or_source_failure')
    h=PrivateHandoff(mode='frozen_pau',proposal_sha256=fingerprint(p.model_dump()),corpus_fingerprint=CORPUS,documents=docs,cases=cases)
    validate_handoff(h.model_dump(),p,passages,questions,docs)
    durable_json(out/'handoff.json',h.model_dump())
    durable_json(out/'retrieval.json',{'model_load_seconds':load,'queries':timings,
        'process_lifetime_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'handoff_sha256':sha(out/'handoff.json')})
    # Producer process exits after return. Parent waits for exit before consumer starts.


def consumer(root,out):
    p=proposal(root);docs,passages,questions=inputs(root,p)
    meta=json.loads((out/'retrieval.json').read_text())
    require(sha(out/'handoff.json')==meta['handoff_sha256'],'source_validation_failure')
    h=validate_handoff(json.loads((out/'handoff.json').read_text()),p,passages,questions,docs)
    pre={'ordering':'fixed_direct_first','source_validation':'pass','reference_scoring_enabled':False}
    durable_json(out/'preflight.json',pre)
    pre['resources']=resources()
    from transformers import AutoTokenizer
    directory=root/'data/models/qwen3-4b-mlx'/p.model_revision
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    token_lists={};counts=[]
    for case in h.cases:
        e=pack(case,h,p)
        for arm,messages in [('direct',build_direct_prompt(case.question)),('rag',rag_prompt(case.question,e))]:
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
            if not isinstance(ids,list) or not ids or not all(type(i) is int for i in ids):raise Halt('tokenization_failure')
            counts.append({'case_id':case.case_id,'arm':arm,'input_tokens':len(ids)})
            token_lists[(case.case_id,arm)]=ids
    pre['prompt_tokens']=counts
    if any(x['input_tokens']>7808 for x in counts):
        pre.update(status='fail',reason='context_overflow');durable_json(out/'preflight.json',pre);raise Halt('context_overflow')
    # Test durable reservation/commit infrastructure without consuming any model slot.
    probe=AttemptJournal(out/'journal-probe',[('probe','no_decode')]);probe.before('probe','no_decode')
    probe.after({'case_id':'probe','arm':'no_decode','status':'no_model_call'})
    pre['journal']='pass';pre['all_24_prompt_budgets']='pass';durable_json(out/'preflight.json',pre)
    from ..rag.two_stage import MLXTextBackend
    config=json.loads((root/p.mlx_config_path).read_text())
    backend=MLXTextBackend(directory,config,max_tokens=384);decoder=QwenDecoder(backend,p)
    if backend.load_peak_bytes>backend.memory_limit_bytes:raise Halt('resource_limit')
    # Match every loaded backend template exactly before the first decode.
    for case in h.cases:
        e=pack(case,h,p)
        for arm,messages in [('direct',build_direct_prompt(case.question)),('rag',rag_prompt(case.question,e))]:
            if decoder.prompt_tokens(messages)!=token_lists[(case.case_id,arm)]:raise Halt('tokenization_mismatch')
    pre['resources_after_load']=resources(1536*1024**2)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>8*1024**3:raise Halt('resource_limit')
    pre.update(status='pass',model_load_seconds=backend.load_seconds,load_mlx_peak_bytes=backend.load_peak_bytes,
               load_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               telemetry='Existing backend exposes token/hash/decode/peak/RSS fields; missing runtime values halt subsequent calls')
    durable_json(out/'preflight.json',pre)
    journal=AttemptJournal(out/'attempts',[(c.case_id,a) for c in p.cases for a in ('direct','rag')])
    runner=PairedComparison(decoder,p,journal=journal)
    try:
        for case in h.cases:
            # Revalidate complete documents and selected records before each pair.
            d,ps,qs=inputs(root,p)
            validate_handoff(h.model_dump(),p,ps,qs,d)
            if sha(out/'handoff.json')!=meta['handoff_sha256']:raise Halt('source_validation_failure')
            resources(1536*1024**2)
            runner.pair(case.case_id,case.question,pack(case,h,p),docs,frozen_passages=passages)
            durable_json(out/'results.json',runner.result())
            if runner.halted or any(r['status']=='blocked' for r in runner.result()['rows']):
                reason=next((r['failure_category'] for r in reversed(runner.result()['rows']) if r['failure_category']), 'runtime_halt')
                raise Halt(reason)
    finally:
        durable_json(out/'results.json',runner.result())


def worker(root,out,stage):
    marker=json.loads((root/'data'/MARKER).read_text())
    require(marker['directory']==str(out) and marker['approval_sha256']==APPROVAL_SHA,'execution_not_authorized')
    require(out.resolve()==out and out.stat().st_mode & 0o777==0o700,'private_directory_failure')
    claim=AttemptJournal(out/'stages',[(stage,'once')]);claim.before(stage,'once')
    try:
        if stage=='resource-probe':durable_json(out/'resource-probe.json',resources())
        elif stage=='producer':producer(root,out)
        elif stage=='consumer':consumer(root,out)
        else:raise Halt('unknown_stage')
        claim.after({'case_id':stage,'arm':'once','status':'completed'})
        return 0
    except BaseException as exc:
        code=getattr(exc,'code','source_validation_failure' if isinstance(exc,ValueError) else 'runtime_or_resource_failure')
        durable_json(out/'halt.json',{'stage':stage,'reason':code,'retry_permitted':False})
        claim.after({'case_id':stage,'arm':'once','status':'halted','reason':code})
        return 1


def execute(root,out,approval_file):
    root=Path(root).resolve();out=Path(out).absolute()
    require(sha(approval_file)==APPROVAL_SHA,'Approval request does not match this bounded run')
    out=private_directory(root,out)
    # One durable root-level authorization consumption, independent of output directory.
    from .qa_runtime import private_write
    private_write(root/'data'/MARKER,{'directory':str(out),'approval_sha256':APPROVAL_SHA,'execution':'one_run_no_retry'})
    durable_json(out/'authorization.json',{'approval_sha256':APPROVAL_SHA,'max_retrievals':12,'max_decodes':24,
        'max_new_tokens_per_call':384,'fixed_order':'direct_first','reference_scoring_enabled':False})
    try:
        durable_json(out/'read-only-preflight.json',check_local(root))
    except BaseException:
        durable_json(out/'halt.json',{'stage':'read_only_preflight','reason':'source_or_configuration_validation_failure','retry_permitted':False})
        return 1
    for stage,env in [('resource-probe','.venv-rag'),('producer','.venv'),('consumer','.venv-rag')]:
        # Blocking wait ensures producer resources are released before consumer loading.
        proc=subprocess.run([str(root/env/'bin/python'),str(root/'scripts/qa_real_worker.py'),stage,str(out)],cwd=root,
            env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1',
                 'TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'},capture_output=True)
        if proc.returncode:
            if not (out/'halt.json').exists():durable_json(out/'halt.json',{'stage':stage,'reason':'worker_interrupted_or_runtime_failure','retry_permitted':False})
            return 1
    durable_json(out/'complete.json',{'status':'completed_once','reference_scoring_enabled':False,'retry_permitted':False})
    return 0
