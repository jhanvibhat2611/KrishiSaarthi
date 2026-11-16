"""Preparation-only local orchestration. Real execution is deliberately unavailable."""
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Literal
from .qa_comparison_fake import StrictModel, Question, fingerprint
from .qa_readiness import E5ComparisonProposal
from .qa_adapters import PairedComparison, require
from ..schemas import Document, Passage, Query, SourceMetadata
from ..ingestion.storage import read_jsonl
from ..evaluation.labels import validate_corpus
from ..rag.models import EvidencePack

CORPUS = 'a919560be507b862be119a236fd1d2501f8d7c32d8fe6ea4951114b205f8c88a'
PASSAGES = '757436adcc6bb0244f92daaac56888637bd679ae2bcbd4e8dc5c8d518ebb6125'
DOCUMENTS_SHA = '5331ad2bd93bf157a9d59e7f27a9725948db0198e35169e43b931c88ff4bc0ac'
QUERIES_SHA = 'de070667195ac999c864367f18efba094e429c809d1715307c1795d37b924ec5'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def proposal(root):
    return E5ComparisonProposal.model_validate_json((root/'configs/qa-readiness-e5.proposed.json').read_text(),strict=True)


def environment(python, packages):
    code = "import importlib.metadata as m,json,sys,platform; names=json.loads(sys.argv[1]); print(json.dumps({'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),'packages':{n:m.version(n) for n in names}}))"
    result = subprocess.run([str(python),'-c',code,json.dumps(packages)],capture_output=True,text=True,
                            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=30)
    require(result.returncode==0,'Environment metadata check failed')
    info=json.loads(result.stdout)
    require(info['packages']==packages,'Installed dependency versions differ from pinned configuration')
    return info


def check_local(root):
    """Read-only validation; no encoders, tokenizers, model constructors or labels."""
    root=Path(root).resolve();p=proposal(root)
    require(p.corpus_fingerprint==CORPUS,'Wrong corpus')
    c=root/'data/pilot-v0.1'
    for name,digest in json.loads((c/'integrity.json').read_text()).items():
        f=(c/name).resolve()
        require(f.is_relative_to(c.resolve()) and sha(f)==digest,'Frozen ledger mismatch')
    require(sha(c/'documents.jsonl')==DOCUMENTS_SHA,'Frozen complete document record mismatch')
    docs=read_jsonl(c/'documents.jsonl',Document);passages=read_jsonl(c/'passages.jsonl',Passage)
    require(len(docs)==1 and docs[0].document_id=='pau-rabi-2025-26' and len(passages)==15,'Wrong source collection')
    require(fingerprint([x.model_dump() for x in passages])==PASSAGES,'Frozen passage mismatch')
    validate_corpus(docs,passages)
    qfile=root/'data/experiments/bm25-pilot-v0.1-run-001/queries.jsonl'
    require(sha(qfile)==QUERIES_SHA,'Signed query snapshot mismatch')
    all_queries=read_jsonl(qfile,Query)
    require(len(all_queries)==48 and len({q.query_id for q in all_queries})==48,'Invalid signed query collection')
    byid={q.query_id:q for q in all_queries}
    for case in p.cases:
        require(case.case_id in byid and byid[case.case_id].language==case.language,'Missing development variant')
        Question(text=byid[case.case_id].text,language=case.language)
    spec=json.loads((root/'configs/dense-models.json').read_text())['e5']
    require(spec['revision']==p.retrieval_revision and spec['tokenizer_revision']==p.retrieval_tokenizer_revision
            and spec['model_id']==p.retrieval_model_id,'E5 identifier mismatch')
    weights=root/'data/models/e5'/spec['revision']
    downloaded=json.loads((weights/'download-manifest.json').read_text())
    require(downloaded['model']==spec,'E5 checkpoint manifest mismatch')
    for name,digest in downloaded['files_sha256'].items():
        f=(weights/name).resolve()
        require(f.is_relative_to(weights.resolve()) and sha(f)==digest,'E5 checkpoint bytes mismatch')
    cache=(root/p.index_manifest_path).resolve()
    require(cache.is_relative_to((root/'data/embeddings').resolve()) and cache.parent.name==p.embedding_cache_key
            and sha(cache)==p.embedding_manifest_sha256,'Embedding manifest mismatch')
    meta=json.loads(cache.read_text())
    require(meta['binding']=={'model_fingerprint':p.retrieval_model_fingerprint,'corpus_fingerprint':CORPUS,
        'passages':fingerprint([x.model_dump() for x in sorted(passages,key=lambda x:x.passage_id)])},'Embedding source binding mismatch')
    require(sha(cache.parent/'embeddings.npy')==meta['array_sha256'],'Embedding bytes mismatch')
    mlx_config=root/p.mlx_config_path
    require(sha(mlx_config)==p.mlx_config_sha256,'Qwen configuration mismatch')
    config=json.loads(mlx_config.read_text())
    # Existing verifier hashes bytes only; importing it never initializes MLX.
    from ..rag.mlx_generator import validate_checkpoint
    validate_checkpoint(root/'data/models/qwen3-4b-mlx'/p.model_revision,config)
    # Historical inference metadata supplies exact existing E5 package pins, not rankings/labels.
    e5=json.loads((root/'data/experiments/e5-pilot-v0.1-run-001/manifest.json').read_text())
    require(e5['model_fingerprint']==p.retrieval_model_fingerprint,'E5 inference identity mismatch')
    require(fingerprint({'spec':e5['model'],'inference':e5['inference']})==p.retrieval_model_fingerprint,'E5 configuration binding mismatch')
    environments={'e5':environment(root/'.venv/bin/python',e5['inference']['versions']),
                  'mlx':environment(root/'.venv-rag/bin/python',p.generation_packages)}
    return {'status':'metadata_and_source_checks_passed','corpus_fingerprint':CORPUS,'passages':15,
            'development_variants':12,'signed_queries_sha256':QUERIES_SHA,'environments':environments,
            'real_execution_enabled':False,'model_loads':0,'model_calls':0,'retrieval_calls':0,
            'runtime_limitations':['Metal availability, real template lengths and peak memory are not tested']}


class HandoffCase(StrictModel):
    case_id: str
    question: Question
    passages: list[Passage]


class PrivateHandoff(StrictModel):
    schema_version: Literal[1]=1
    mode: Literal['synthetic','frozen_pau']
    proposal_sha256: str
    corpus_fingerprint: str
    documents: list[Document]
    cases: list[HandoffCase]


def validate_complete_documents(actual, expected):
    require(expected is not None and [d.model_dump() for d in actual]==[d.model_dump() for d in expected],
            'Complete document metadata mismatch')


def validate_handoff(payload, p, frozen_passages=None, expected_questions=None, expected_documents=None):
    h=PrivateHandoff.model_validate(payload,strict=True)
    require(h.proposal_sha256==fingerprint(p.model_dump()),'Handoff proposal mismatch')
    require([x.case_id for x in h.cases]==[x.case_id for x in p.cases],'Handoff development matrix mismatch')
    if h.mode=='frozen_pau':
        require(h.corpus_fingerprint==CORPUS and frozen_passages is not None
                and fingerprint([x.model_dump() for x in frozen_passages])==PASSAGES,'Frozen handoff binding mismatch')
        require(expected_questions is not None and set(expected_questions)=={c.case_id for c in p.cases},'Signed question binding required')
        validate_complete_documents(h.documents,expected_documents)
        allowed={x.passage_id:x.model_dump() for x in frozen_passages}
        require(len(h.documents)==1 and h.documents[0].document_id=='pau-rabi-2025-26','Wrong source')
    else:
        require(h.corpus_fingerprint=='synthetic-only','Synthetic mode fingerprint mismatch')
        require(all(d.document_id.startswith('synthetic-') and d.canonical_text.startswith('SYNTHETIC:') for d in h.documents),'Not synthetic sources')
    for c,expected in zip(h.cases,p.cases):
        require(c.question.language==expected.language and len(c.passages)==5
                and len({x.passage_id for x in c.passages})==5,'Invalid question or Top-5 handoff')
        if h.mode=='frozen_pau':
            require(c.question==expected_questions[c.case_id],'Question text differs from signed variant')
            require(all(allowed.get(x.passage_id)==x.model_dump() for x in c.passages),'Unapproved evidence in handoff')
        else:
            require(all(x.document_id.startswith('synthetic-') and x.text.startswith('SYNTHETIC:') for x in c.passages),'Not synthetic passages')
        validate_corpus(h.documents,c.passages)
    return h


def private_write(path, payload):
    data=json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False).encode()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    from .qa_journal import fsync_directory
    fsync_directory(Path(path).parent)


def private_directory(root, directory):
    root=Path(root).resolve();directory=Path(directory).absolute()
    require(directory.resolve()==directory and directory.is_relative_to(root/'data'),'Use a private directory inside data/')
    require(subprocess.run(['git','check-ignore','--quiet',str(directory/'handoff.json')],cwd=root,capture_output=True).returncode==0,'Output must be ignored')
    directory.mkdir(mode=0o700,parents=False,exist_ok=False)
    return directory


def consume_synthetic(root, directory, expected_sha):
    root=Path(root).resolve();directory=Path(directory).absolute();f=directory/'handoff.json'
    require(directory.resolve()==directory and directory.is_relative_to(root/'data'),'Private handoff path required')
    require(stat.S_IMODE(directory.stat().st_mode)==0o700 and stat.S_IMODE(f.stat().st_mode)==0o600,'Private permissions required')
    require(not f.is_symlink() and sha(f)==expected_sha,'Handoff checksum mismatch')
    p=proposal(root);h=validate_handoff(json.loads(f.read_text()),p)
    require(h.mode=='synthetic','Real handoff consumption is disabled')
    decoder=SyntheticDecoder();runner=PairedComparison(decoder,p)
    try:
        for case in h.cases:
            evidence=EvidencePack(question=Query(query_id=case.case_id,**case.question.model_dump()),
                passages=case.passages,retriever_id='synthetic-only',
                document_metadata={d.document_id:d.domain_metadata for d in h.documents})
            runner.pair(case.case_id,case.question,evidence,h.documents)
    finally:
        result=runner.result()
        result['notice']='Synthetic development-matrix rehearsal; invented questions/outputs, not agricultural results.'
        result['token_counts_kind']='Synthetic character-code tokenizer, not Qwen token measurements'
        private_write(directory/'mock-results.json',result)
    return {'mode':'synthetic','planned':24,'mock_decode_calls':len(decoder.calls),'model_calls':0,'retrieval_calls':0,
            'failed':result['overall']['failed'],'withholding':result['overall']['withholding'],
            'unattempted':result['overall']['unattempted'],
            'blocked_slots':sum(r['status']=='blocked' for r in result['rows'])}


class SyntheticDecoder:
    execution_kind='mock'
    def __init__(self): self.calls=[];self.last_output_text='';self.last_metrics={}
    def prompt_tokens(self,messages): return list(json.dumps(messages,ensure_ascii=False).encode())
    def decode(self,messages,max_tokens):
        require(max_tokens==384,'Wrong output cap')
        self.calls.append(messages)
        rag='sources' in json.loads(messages[1]['content'])
        raw=json.dumps({'status':'draft','claims':[{'text':'Invented tag Cedar.',**({'evidence_handles':['E1']} if rag else {})}],'reason':'Synthetic fixture'})
        if len(self.calls)==3: raw='{deliberately malformed synthetic JSON'
        if len(self.calls)==4: raw=json.dumps({'status':'uncertain','claims':[],'reason':'Canned uncertainty'})
        self.last_output_text=raw
        self.last_metrics={'input_tokens':len(self.prompt_tokens(messages)),'output_tokens':len(raw.encode()),
                           'runtime_seconds':0.0,'finish_reason':'stop','measurement_kind':'mock_fixture'}
        return raw


def dry_run(root, directory):
    from ..ingestion.plaintext import PlainTextIngestor
    from ..preprocessing.chunking import DeterministicChunker
    from ..config import ChunkingConfig
    root=Path(root).resolve();directory=private_directory(root,directory);p=proposal(root)
    source=directory/'synthetic.txt'
    fd=os.open(source,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:f.write('SYNTHETIC: Invented tag Cedar. No agricultural recommendation.')
    doc=PlainTextIngestor().ingest(source,SourceMetadata(document_id='synthetic-c3'))
    first=DeterministicChunker(ChunkingConfig()).chunk(doc)[0]
    # Five unique synthetic records exercise the Top-5 transport, not retrieval quality.
    passages=[first.model_copy(update={'passage_id':f'synthetic-c3-{i}'}) for i in range(5)]
    questions={'en':'Which fictional tag is shown?','hi':'कौन-सा काल्पनिक टैग दिखाया गया है?',
               'hinglish':'Kaunsa fictional tag dikhaya gaya hai?'}
    h=PrivateHandoff(mode='synthetic',proposal_sha256=fingerprint(p.model_dump()),corpus_fingerprint='synthetic-only',
        documents=[doc],cases=[HandoffCase(case_id=c.case_id,question=Question(text=questions[c.language],language=c.language),passages=passages) for c in p.cases])
    validate_handoff(h.model_dump(),p)
    path=directory/'handoff.json';private_write(path,h.model_dump());digest=sha(path)
    # Consumer runs in the existing MLX environment but imports no model libraries.
    proc=subprocess.run([str(root/'.venv-rag/bin/python'),str(root/'scripts/qa_runtime.py'),
        'consume-synthetic','--directory',str(directory),'--handoff-sha256',digest],cwd=root,
        env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=60)
    require(proc.returncode==0,'Synthetic consumer failed; no protected output logged')
    summary=json.loads(proc.stdout);private_write(directory/'summary.json',summary)
    return summary
