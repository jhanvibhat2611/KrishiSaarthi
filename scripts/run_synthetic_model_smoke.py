"""Six-call offline synthetic smoke. No real corpus or research labels are read."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
from krishisaarthi.ui.real_synthetic import SyntheticSource, messages, generate_once, CONTRACT_V1, CONTRACT_V2, CURRENT_CONTRACT
from krishisaarthi.ui.server import DOCUMENTS

REQUESTS=[
 {'query_id':'smoke-en-supported','language':'en','question':'What are the names of the two fictional soil samples?'},
 {'query_id':'smoke-en-missing','language':'en','question':'On what calendar date were the fictional soil samples collected?'},
 {'query_id':'smoke-hi-supported','language':'hi','question':'दो काल्पनिक मिट्टी के नमूनों के नाम क्या हैं?'},
 {'query_id':'smoke-hi-missing','language':'hi','question':'काल्पनिक मिट्टी के नमूने किस तारीख को एकत्र किए गए थे?'},
 {'query_id':'smoke-hinglish-supported','language':'hinglish','question':'Dono fictional mitti ke samples ke naam kya hain?'},
 {'query_id':'smoke-hinglish-missing','language':'hinglish','question':'Fictional mitti ke samples kis tareekh ko collect kiye gaye the?'},
]

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',required=True)
 parser.add_argument('--model-dir',required=True)
 parser.add_argument('--config',default='configs/mlx-generator.json')
 parser.add_argument('--contract-version',choices=[CONTRACT_V1,CONTRACT_V2],default=CURRENT_CONTRACT)
 args=parser.parse_args();out=Path(args.output)
 out.mkdir(parents=True,exist_ok=False)
 def save(name,value):
  (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 config=json.loads(Path(args.config).read_text())
 raw=next(d for d in DOCUMENTS if d['passage_id']=='synthetic-soil-02')
 sources=[SyntheticSource(**{k:raw[k] for k in ['passage_id','document_id','text','section']})]
 prompts=[messages(r['question'],r['language'],sources,args.contract_version)[0] for r in REQUESTS]
 manifest={'contract_version':args.contract_version,'created_at':datetime.now(timezone.utc).isoformat(),'scope':'Engineering synthetic smoke only; no benchmark outcomes','model':config,'max_tokens_per_call':256,'maximum_calls':6,'batch_size':1,'repairs':0,'retries':0,'requests':REQUESTS,'prompts':prompts,'sources':[s.model_dump() for s in sources],
 'packages':{p:version(p) for p in config['packages']},'code_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['src/krishisaarthi/ui/real_synthetic.py','src/krishisaarthi/rag/two_stage.py','src/krishisaarthi/rag/mlx_generator.py','scripts/run_synthetic_model_smoke.py']}}
 save('frozen-manifest.json',manifest)
 manifest_hash=hashlib.sha256((out/'frozen-manifest.json').read_bytes()).hexdigest()
 # Hash validation is performed by the existing loader before MLX model loading.
 from krishisaarthi.rag.two_stage import MLXTextBackend
 try:
  backend=MLXTextBackend(args.model_dir,config)
 except Exception as exc:
  save('completion.json',{'contract_version':args.contract_version,'calls_attempted':0,'stop_reason':'integrity_load_or_resource_failure','exception_type':type(exc).__name__,'manifest_sha256':manifest_hash})
  raise
 save('load.json',{'seconds':backend.load_seconds,'mlx_peak_bytes':backend.load_peak_bytes,'allocation_limit_bytes':backend.memory_limit_bytes,'cache_limit_bytes':256*1024**2})
 completed=[];stop=None
 for request,prompt in zip(REQUESTS,prompts):
  if hashlib.sha256((out/'frozen-manifest.json').read_bytes()).hexdigest()!=manifest_hash:raise ValueError('Frozen manifest changed')
  if messages(request['question'],request['language'],sources,args.contract_version)[0]!=prompt:raise ValueError('Prompt changed')
  result=generate_once(backend,request,sources,256,contract_version=args.contract_version);completed.append(result)
  save(f"{request['query_id']}.json",result)
  response=result['response'];print(request['query_id'],response['status'],response['failure_category'],flush=True)
  if response['failure_category']=='model_runtime_error' or result['metrics'].get('mlx_peak_bytes',0)>backend.memory_limit_bytes:
   stop='runtime_or_resource_failure';break
 save('completion.json',{'contract_version':args.contract_version,'calls_attempted':len(completed),'stop_reason':stop,'manifest_sha256':manifest_hash,'model_loaded_once':True,'no_retries':True,'results':[r['response'] for r in completed]})

if __name__=='__main__':main()
