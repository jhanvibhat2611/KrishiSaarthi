"""Offline exploratory draft run; input is questions only, never gold labels."""
import argparse,hashlib,json,os,platform,subprocess,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from krishisaarthi.schemas import Document,Passage,Query
from krishisaarthi.ingestion.storage import read_jsonl
from krishisaarthi.retrieval.bm25 import BM25Retriever
from krishisaarthi.rag import EvidenceSelector,RAGPipeline
from krishisaarthi.rag.mlx_generator import MLXGenerator
from krishisaarthi.validation import require


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--queries',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--corpus',type=Path,default=Path('data/pilot-v0.1'))
    p.add_argument('--model-config',type=Path,default=Path('configs/mlx-generator.json'))
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--max-tokens',type=int,default=512)
    args=p.parse_args();root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    require(out.is_relative_to(root/'data') and not out.exists(),'Use fresh Git-ignored output')
    docs=read_jsonl(args.corpus/'documents.jsonl',Document);passages=read_jsonl(args.corpus/'passages.jsonl',Passage)
    queries=read_jsonl(args.queries,Query)
    input_hashes={str(f.resolve()):sha(f) for f in [args.corpus/'documents.jsonl',args.corpus/'passages.jsonl',args.queries,args.model_config]}
    selector=EvidenceSelector(BM25Retriever(),docs,passages)
    cfg=json.loads(args.model_config.read_text());out.mkdir(parents=True)
    def save(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    # Stop on model-load failure; do not silently replace runtime or checkpoint.
    generator=MLXGenerator(args.model_dir,cfg,args.max_tokens)
    print('Loaded model:',generator.load_seconds,'seconds; MLX peak bytes',generator.load_peak_bytes,flush=True)
    pipeline=RAGPipeline(selector,generator);records=[]
    for q in queries:
        pack=selector.select(q,5);prompt=pipeline.prompt_builder.build(pack)
        tokens=generator.tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True)
        print(q.query_id,'prompt tokens',len(tokens),flush=True)
        require(len(tokens)+args.max_tokens<=cfg['inference']['context_limit'],'Prompt budget exceeded; no truncation')
        result=pipeline.run(q,k=5,mode='research_draft')
        row={'query':q.model_dump(),'evidence':pack.model_dump(),'prompt':prompt,'response':result.model_dump(),
            'raw_generator_text':generator.last_output_text,'independent_human_verification':False}
        records.append(row);save('results.json',records)
        print(q.query_id,result.status,generator.last_metrics,flush=True)
        if generator.last_metrics.get('mlx_peak_bytes',0)>6*1024**3:raise RuntimeError('Memory safety limit exceeded; stopping')
        if result.status=='generator_error' and any('Memory' in w for w in result.warnings):raise RuntimeError('Memory failure; stopping')
    require(all(sha(Path(n))==h for n,h in input_hashes.items()),'Inputs changed')
    save('manifest.json',{'mode':'research_draft','retriever':'BM25 fixed defaults; top5; no translation or query-specific routing',
        'input_sha256':input_hashes,'model':cfg,'max_tokens':args.max_tokens,'load_seconds':generator.load_seconds,
        'load_peak_mlx_bytes':generator.load_peak_bytes,'python':platform.python_version(),
        'source_sha256':{str(f.relative_to(root)):sha(f) for f in (root/'src/krishisaarthi/rag').glob('*.py')},
        'script_sha256':sha(Path(__file__)),'packages':subprocess.check_output([str(root/'.venv-rag/bin/python'),'-m','pip','freeze'],text=True),
        'note':'Exploratory generation, not independent factual verification. No annotations or gold labels used at inference.'})
    save('integrity.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})

if __name__=='__main__':main()
