"""Download only pinned files from the two approved checkpoints; no datasets/heads."""
from krishisaarthi.validation import require
import os
os.environ['HF_HUB_DISABLE_XET']='1'
import argparse, hashlib,json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=['e5','bge-m3']);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];config=json.loads((root/'configs/dense-models.json').read_text())[args.name]
    require(config['model_id'] in ['intfloat/multilingual-e5-base','BAAI/bge-m3'], 'download_dense_models.py:11: integrity validation failed')
    revision=config['revision'];target=root/'data/models'/args.name/revision
    files=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model',config['weight_file'],'README.md']
    info=HfApi().model_info(config['model_id'],revision=revision,files_metadata=True)
    require(info.sha==revision, 'download_dense_models.py:15: integrity validation failed')
    snapshot_download(config['model_id'],revision=revision,allow_patterns=files,local_dir=target,max_workers=1)
    hashes={}
    for filename in files:
        p=target/filename;h=hashlib.sha256(p.read_bytes()).hexdigest();hashes[filename]=h
        meta=next(s for s in info.siblings if s.rfilename==filename)
        if meta.lfs:require(h==meta.lfs.sha256, f'LFS hash mismatch: {filename}')
    record={'model':config,'files_sha256':hashes,'approved_files_only':True}
    (target/'download-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'model':config['model_id'],'revision':revision,'path':str(target),'files_verified':len(hashes)}),flush=True)
