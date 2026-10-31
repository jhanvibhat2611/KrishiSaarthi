"""Optional, strictly local MLX adapter. No download function or implicit Hub load."""
import hashlib
import json
import platform
import resource
import time
from importlib.metadata import version
from pathlib import Path
from ..validation import require
from .models import GeneratedAnswer


def validate_checkpoint(directory, config):
    directory=Path(directory).resolve()
    require(directory.is_dir(),'Approved local checkpoint directory is missing; downloads are not performed')
    require(config['model_id']=='mlx-community/Qwen3-4B-Instruct-2507-4bit','Unexpected generator model')
    require(config['revision']==config['tokenizer_revision'] and len(config['revision'])==40,'Pinned tokenizer revision required')
    for name,spec in config['files'].items():
        p=(directory/name).resolve()
        require(p.is_relative_to(directory) and p.is_file(),'Missing local checkpoint file: '+name)
        require(p.stat().st_size==spec['size'],'Checkpoint size mismatch: '+name)
        if 'sha256' in spec:
            with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            require(digest==spec['sha256'],'Checkpoint SHA256 mismatch: '+name)
        else:
            data=p.read_bytes();digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            require(digest==spec['git_blob_sha1'],'Pinned tokenizer/config content mismatch: '+name)
    return directory

class MLXGenerator:
    def __init__(self,directory,config,max_tokens=None):
        directory=validate_checkpoint(directory,config)
        require(platform.system()=='Darwin' and platform.machine()=='arm64','MLX adapter requires Apple Silicon')
        for package,pin in config['packages'].items():
            require(version(package)==pin,'Install pinned optional package '+package+'=='+pin)
        self.config=config;self.max_tokens=max_tokens if max_tokens is not None else config['inference']['max_tokens']
        require(0<self.max_tokens<config['inference']['context_limit'],'Invalid generation length')
        import mlx.core as mx
        from mlx_lm import load
        self.mx=mx
        require(mx.metal.is_available(), 'Apple Metal device unavailable; model not loaded')
        self.memory_limit_bytes=6*1024**3
        mx.set_memory_limit(self.memory_limit_bytes)
        mx.set_cache_limit(256*1024**2)
        mx.reset_peak_memory();t=time.perf_counter()
        self.model,self.tokenizer=load(str(directory),tokenizer_config={'local_files_only':True,'trust_remote_code':False})
        mx.eval(self.model.parameters())
        self.load_seconds=time.perf_counter()-t;self.load_peak_bytes=mx.get_peak_memory()
        self.last_metrics={}

    def generate(self,messages,evidence):
        self.last_metrics = {}
        self.last_output_text = ""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        require(len(prompt)+self.max_tokens<=self.config['inference']['context_limit'],'Prompt plus output exceeds configured context budget; no truncation')
        self.mx.random.seed(self.config['inference']['seed']);self.mx.reset_peak_memory();start=time.perf_counter()
        parts=[];tokens=0
        try:
            for response in stream_generate(self.model,self.tokenizer,prompt=prompt,
                    max_tokens=self.max_tokens,sampler=make_sampler(temp=0.0),
                    prefill_step_size=self.config['inference']['prompt_prefill_step_size']):
                parts.append(response.text);tokens+=1
            self.last_output_text=''.join(parts)
            result=GeneratedAnswer.model_validate_json(self.last_output_text)
        finally:
            self.last_metrics={'runtime_seconds':time.perf_counter()-start,
                'mlx_peak_bytes':self.mx.get_peak_memory(),'process_lifetime_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'memory_limit_bytes':getattr(self,'memory_limit_bytes',None),'load_seconds':self.load_seconds,'load_peak_mlx_bytes':self.load_peak_bytes,
                'input_tokens':len(prompt),'output_steps':tokens,'output_tokens':getattr(response,'generation_tokens',tokens) if parts else 0,'max_tokens':self.max_tokens,
                'model_revision':self.config['revision'],'temperature':0.0,'seed':self.config['inference']['seed'],
                'prompt_tokens_sha256':hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                'determinism_note':'Greedy decoding with fixed seed; cross-hardware bitwise reproducibility is not guaranteed.'}
        return result
