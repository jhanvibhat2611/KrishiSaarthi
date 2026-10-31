"""Local-only CLI. Does not read annotation files or gold labels."""
import argparse
import json
from pathlib import Path
from ..schemas import Document, Passage, Query
from ..ingestion.storage import read_jsonl
from ..retrieval.bm25 import BM25Retriever
from . import EvidenceSelector, MockGenerator, RAGPipeline
from .models import SufficiencyReview


def main():
    p=argparse.ArgumentParser(description='Conservative local RAG; no automatic sufficiency approval')
    p.add_argument('--documents',type=Path,required=True);p.add_argument('--passages',type=Path,required=True)
    p.add_argument('--question',required=True);p.add_argument('--k',type=int,default=5)
    p.add_argument('--review',type=Path,help='Fresh human sufficiency review bound to exact question/evidence fingerprint')
    p.add_argument('--evidence-only',action='store_true',help='Show evidence pack for review, without generation')
    p.add_argument('--mode',choices=['conservative','research_draft'],default='conservative')
    p.add_argument('--generator',choices=['mock','mlx'],default='mock')
    p.add_argument('--model-dir',type=Path);p.add_argument('--model-config',type=Path,default=Path('configs/mlx-generator.json'))
    p.add_argument('--max-tokens',type=int)
    args=p.parse_args()
    selector=EvidenceSelector(BM25Retriever(),read_jsonl(args.documents,Document),read_jsonl(args.passages,Passage))
    q=Query(query_id='interactive',text=args.question)
    if args.evidence_only:
        pack=selector.select(q,args.k)
        print(json.dumps({'evidence_fingerprint':pack.fingerprint,'evidence':pack.model_dump(),
            'review_template':{'evidence_fingerprint':pack.fingerprint,'reviewer_id':None,'reviewed_at':None,
            'all_question_components_covered':False,'applicability_checked':False,'components':[],
            'unresolved_concerns':['Pending actual human review']}},ensure_ascii=False,indent=2));return
    review=SufficiencyReview.model_validate_json(args.review.read_text()) if args.review else None
    generator=MockGenerator()
    if args.generator=='mlx':
        if args.model_dir is None:p.error('--model-dir is required; no automatic downloads')
        from .mlx_generator import MLXGenerator
        generator=MLXGenerator(args.model_dir,json.loads(args.model_config.read_text()),args.max_tokens)
    result=RAGPipeline(selector,generator).run(q,args.k,review,mode=args.mode)
    print(result.model_dump_json(indent=2))

if __name__=='__main__':main()
