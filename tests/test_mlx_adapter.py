"""Fake MLX runtime and tiny synthetic files; no weights or optional imports needed."""
import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
from krishisaarthi.rag.mlx_generator import validate_checkpoint, MLXGenerator


def test_checkpoint_content_checks(tmp_path):
    content=b'synthetic';(tmp_path/'tokenizer.json').write_bytes(content)
    config={'model_id':'mlx-community/Qwen3-4B-Instruct-2507-4bit','revision':'a'*40,'tokenizer_revision':'a'*40,
        'files':{'tokenizer.json':{'size':len(content),'git_blob_sha1':hashlib.sha1(b'blob 9\0'+content).hexdigest()}}}
    assert validate_checkpoint(tmp_path,config)==tmp_path
    (tmp_path/'tokenizer.json').write_bytes(b'TAMPERED!')
    with pytest.raises(ValueError,match='content mismatch'):validate_checkpoint(tmp_path,config)
    (tmp_path/'tokenizer.json').unlink()
    with pytest.raises(ValueError,match='Missing'):validate_checkpoint(tmp_path,config)

def test_weight_hash_rejection(tmp_path):
    (tmp_path/'model.safetensors').write_bytes(b'fake')
    config={'model_id':'mlx-community/Qwen3-4B-Instruct-2507-4bit','revision':'a'*40,'tokenizer_revision':'a'*40,
        'files':{'model.safetensors':{'size':4,'sha256':'0'*64}}}
    with pytest.raises(ValueError,match='SHA256'):validate_checkpoint(tmp_path,config)

def test_adapter_missing_checkpoint_never_imports_mlx(tmp_path):
    with pytest.raises(ValueError,match='missing'):MLXGenerator(tmp_path/'absent',{})

def test_fake_runtime_prompt_and_metrics(monkeypatch):
    seen={}
    def stream(*args,**kwargs):
        seen.update(kwargs)
        yield SimpleNamespace(text='{"claims":[],"missing_context":["unavailable"],"abstain":true}')
    def sampler(**kw):
        assert kw=={'temp':0.0};return 'greedy'
    monkeypatch.setitem(sys.modules,'mlx_lm',SimpleNamespace(stream_generate=stream))
    monkeypatch.setitem(sys.modules,'mlx_lm.sample_utils',SimpleNamespace(make_sampler=sampler))
    def template(messages,**kw):
        assert kw=={'tokenize':True,'add_generation_prompt':True};return [1,2,3]
    g=MLXGenerator.__new__(MLXGenerator)
    g.tokenizer=SimpleNamespace(apply_chat_template=template);g.model='fake'
    g.mx=SimpleNamespace(random=SimpleNamespace(seed=lambda s:None),reset_peak_memory=lambda:None,get_peak_memory=lambda:123)
    g.config={'revision':'a'*40,'inference':{'context_limit':20,'seed':0,'prompt_prefill_step_size':256}}
    g.max_tokens=10;g.load_seconds=.01;g.load_peak_bytes=100
    assert g.generate([{'role':'user','content':'synthetic'}],None).abstain
    assert seen['prompt']==[1,2,3] and seen['sampler']=='greedy'
    assert g.last_metrics['mlx_peak_bytes']==123 and g.last_metrics['runtime_seconds']>=0
    g.max_tokens=20
    with pytest.raises(ValueError,match='no truncation'):g.generate([],None)
