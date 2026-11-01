"""Two-stage offline retrieval/generation experiment; questions only, no gold reads."""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
from krishisaarthi.validation import require
from krishisaarthi.config import fingerprint
from krishisaarthi.schemas import Document, Passage, Query, RetrievalResult
from krishisaarthi.ingestion.storage import read_jsonl
from krishisaarthi.rag import EvidenceSelector, RAGPipeline
from krishisaarthi.rag.handles import handle_table
from krishisaarthi.retrieval.bm25 import BM25Retriever

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

class RankingSnapshot:
    """Exact query/corpus-bound ranking replay avoids co-loading encoder and generator."""
    def __init__(self, snapshot, name):
        self.snapshot=snapshot;self.name=name;self.retriever_id=snapshot['retrievers'][name]['retriever_id']
    def index(self,passages):
        require(fingerprint([p.model_dump() for p in passages])==self.snapshot['passages_fingerprint'],'Snapshot corpus mismatch')
    def search(self,q,k):
        row=self.snapshot['retrievers'][self.name]['queries'][q.query_id]
        require(row['query']==q.model_dump(),'Snapshot query mismatch')
        return [RetrievalResult.model_validate(h) for h in row['ranking'][:k]]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate'])
    parser.add_argument('--queries',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--two-stage',action='store_true');parser.add_argument('--rankings',type=Path);parser.add_argument('--model-dir',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(ROOT/'data') and not out.exists(),'Fresh ignored output required')
    c=ROOT/'data/pilot-v0.1';docs=read_jsonl(c/'documents.jsonl',Document);passages=read_jsonl(c/'passages.jsonl',Passage)
    queries=read_jsonl(args.queries,Query)
    for name,h in json.loads((c/'integrity.json').read_text()).items():require(sha(c/name)==h,'Frozen input mismatch')
    inputs={str(p.resolve()):sha(p) for p in [c/'documents.jsonl',c/'passages.jsonl',c/'manifest.json',args.queries]}
    out.mkdir(parents=True)
    if args.stage=='prepare':
        from krishisaarthi.retrieval.dense import HFEncoder
        from krishisaarthi.retrieval.e5 import E5Retriever
        spec=json.loads((ROOT/'configs/dense-models.json').read_text())['e5']
        directory=ROOT/'data/models/e5'/spec['revision'];downloaded=json.loads((directory/'download-manifest.json').read_text())
        require(downloaded['model']==spec,'E5 checkpoint spec mismatch')
        for n,h in downloaded['files_sha256'].items():require(sha(directory/n)==h,'E5 checkpoint mismatch')
        spec={**spec,'checkpoint_files_fingerprint':fingerprint(downloaded['files_sha256'])}
        started=time.perf_counter();encoder=HFEncoder(directory,spec,device='cpu',batch_size=2)
        dense=E5Retriever(encoder,spec,json.loads((c/'manifest.json').read_text())['corpus_fingerprint'],out/'embeddings')
        dense.preflight(passages,queries);dense.prepare_queries(queries)
        result={'passages_fingerprint':fingerprint([p.model_dump() for p in passages]),'input_sha256':inputs,'retrievers':{}}
        for name,retriever in [('bm25',BM25Retriever()),('e5',dense)]:
            EvidenceSelector(retriever,docs,passages)
            result['retrievers'][name]={'retriever_id':retriever.retriever_id,'queries':{q.query_id:{'query':q.model_dump(),'ranking':[h.model_dump() for h in retriever.search(q,15)]} for q in queries}}
        result['e5']={'spec':spec,'inference':encoder.identity,'token_lengths':dense.token_lengths,'timings':dense.timings,'total_prepare_seconds':time.perf_counter()-started}
        save(out/'rankings.json',result);print('Retrieval preparation complete',flush=True)
    else:
        require(args.rankings is not None and args.model_dir is not None,'Rankings and model directory required')
        snapshot=json.loads(args.rankings.read_text());inputs[str(args.rankings.resolve())]=sha(args.rankings)
        require(snapshot['input_sha256']=={k:v for k,v in inputs.items() if k!=str(args.rankings.resolve())},'Snapshot inputs mismatch')
        from krishisaarthi.rag.mlx_generator import MLXGenerator
        cfgpath=ROOT/'configs/mlx-generator.json';cfg=json.loads(cfgpath.read_text());inputs[str(cfgpath)]=sha(cfgpath)
        if args.two_stage:
            from krishisaarthi.rag.two_stage import MLXTextBackend, TwoStageGenerator
            backend=MLXTextBackend(args.model_dir,cfg,max_tokens=384)
            gen=TwoStageGenerator(backend)
        else:
            backend=MLXGenerator(args.model_dir,cfg,max_tokens=768);gen=backend
        records=[]
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/krishisaarthi/rag').glob('*.py')}
        manifest={'input_sha256':inputs,'source_sha256':source_hashes,'script_sha256':sha(Path(__file__)),'configuration':cfg,'max_tokens':({'A':192,'B':384} if args.two_stage else 768),'top_k':5,'repair_attempts':0,'model_load_seconds':backend.load_seconds,'model_load_peak_bytes':backend.load_peak_bytes,'python':platform.python_version(),'stage':'generation','factual_review':'pending human review','prompt_version':(gen.prompt_version if args.two_stage else 'source-grounded-handles-v1'),'retrieval_separation':'E5 CPU ranking in separate process before Qwen loading'}
        save(out/'manifest.json',manifest)
        for name in ['bm25','e5']:
            selector=EvidenceSelector(RankingSnapshot(snapshot,name),docs,passages);pipeline=RAGPipeline(selector,gen)
            for q in queries:
                pack=selector.select(q,5);prompt=pipeline.prompt_builder.build(pack)
                print(name,q.query_id,'starting',flush=True)
                response=pipeline.run(q,k=5,mode='research_draft')
                records.append({'retriever':name,'query':q.model_dump(),'evidence':pack.model_dump(),'handles':{h:p.model_dump() for h,p in handle_table(pack).items()},'prompt':(None if args.two_stage else prompt),'stage_records':getattr(gen,'last_stages',[]),'raw_generator_text':gen.last_output_text,'response':response.model_dump(),'human_factual_verification':'pending'})
                save(out/'results.json',records)
                print(name,q.query_id,response.status,response.failure_category,gen.last_metrics.get('runtime_seconds'),flush=True)
                if response.failure_category=='model_runtime_error':raise RuntimeError('Runtime failure; stopped for inspection')
                require(gen.last_metrics.get('mlx_peak_bytes',0)<=6*1024**3,'Memory safety cap exceeded')
        require(all(sha(ROOT/p)==h for p,h in source_hashes.items()),'Code changed during experiment')
    require(all(sha(p)==h for p,h in inputs.items()),'Inputs changed during experiment')
    save(out/'integrity.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})

if __name__=='__main__':main()
