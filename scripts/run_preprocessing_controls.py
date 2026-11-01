"""Separate exploratory punctuation-filtered BM25/RRF controls; no dense encoding."""
import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from run_hybrid_pilot import ROOT, CORPUS, require, sha, ledger, RecordedRetriever
from krishisaarthi.config import BM25Config, fingerprint, load_config
from krishisaarthi.schemas import Document, Passage, Query, GoldLabel, EvidenceSpan
from krishisaarthi.ingestion.storage import read_jsonl
from krishisaarthi.evaluation.labels import validate_labels
from krishisaarthi.evaluation.metrics import fully_covered
from krishisaarthi.evaluation.runner import evaluate
from krishisaarthi.retrieval.punctuation_filtered import PunctuationFilteredBM25Retriever, FILTERED_TOKENIZER_ID
from krishisaarthi.retrieval.rrf import RRFRetriever


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,help='Fresh directory below ignored data/')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    require(out.is_relative_to(ROOT/'data') and not out.exists(),'Fresh ignored output required')
    started=datetime.now(timezone.utc).isoformat()
    c=ROOT/'data/pilot-v0.1';a=ROOT/'data/annotation-final-signoff-v1'
    exp=ROOT/'data/experiments';hybrid=exp/'hybrid-comparison-v0.1-run-001'
    dirs={'BM25':exp/'bm25-pilot-v0.1-run-001','E5':exp/'e5-pilot-v0.1-run-001',
          'BGE-M3':exp/'bge-m3-pilot-v0.1-run-001'}
    for d in [c,*dirs.values(),hybrid]:ledger(d)
    roots=[c,a,exp,ROOT/'data/phase3b-review-v1',ROOT/'data/phase4.1-reliability-v1',
           ROOT/'data/phase4.2-controlled-v1',ROOT/'data/phase5-retrieval-v1']
    protected={str(p.relative_to(ROOT)):sha(p) for d in roots for p in d.rglob('*') if p.is_file()}
    cfgpath=ROOT/'configs/punctuation-filtered-bm25.json';spec=json.loads(cfgpath.read_text())
    require(spec['tokenizer']==FILTERED_TOKENIZER_ID and spec['rrf_constant']==60 and
            spec['rrf_weights']==[1,1] and spec['gate_nonpositive_lexical'] is False and
            spec['ranking_depth']=='complete_frozen_corpus' and spec['tie_break']=='passage_id' and
            spec['ks']==[1,3,5], 'Control configuration differs from approved experiment')
    docs=read_jsonl(c/'documents.jsonl',Document);ps=read_jsonl(c/'passages.jsonl',Passage)
    b=dirs['BM25'];qs=read_jsonl(b/'queries.jsonl',Query);gold=read_jsonl(b/'gold.jsonl',GoldLabel)
    meta=json.loads((b/'query-evidence-metadata.json').read_text());byq={r['query_id']:r for r in meta}
    require(len(ps)==15 and len(qs)==48 and len(gold)==48 and sum(g.answerable for g in gold)==36,'Counts')
    require(len({r['group_id'] for r in meta})==16 and byq.keys()=={q.query_id for q in qs},'Group metadata')
    require(not validate_labels(qs,gold,docs,ps)[2],'Unexpected label warnings')
    labels={g.query_id:g for g in gold}
    for q in qs:
        if byq[q.query_id]['group_id'][-2:] in ['08','10']:
            require(byq[q.query_id]['evidence_requirement']=='ALL' and len(labels[q.query_id].evidence_spans)==2,'ALL spans')
    fp=fingerprint({'queries':[q.model_dump() for q in qs],'gold':[g.model_dump() for g in gold],'metadata':meta})
    sign=json.loads((a/'reviewer-signoff.json').read_text())
    require(sign['status']=='signed_off' and sign['final_decision']=='accept','Incomplete signoff')
    cfg=load_config(b/'config.toml');require(cfg.evaluation.ks==spec['ks'],'K settings')
    require(cfg.bm25==BM25Config.model_validate(spec['bm25']),'Scoring parameters changed')
    reports={};details={};replay={}
    for name,d in dirs.items():
        manifest=json.loads((d/'manifest.json').read_text())
        require(manifest['corpus_fingerprint']==CORPUS and manifest['evaluation_set_fingerprint']==fp,'Fingerprints')
        require(manifest['annotation_sha256']==sha(a/'annotation-candidate.json')==sign['annotation_sha256']
                and manifest['signoff_sha256']==sha(a/'reviewer-signoff.json'),'Signed annotation hash')
        reports[name]=json.loads((d/'evaluation.json').read_text())
        details[name]=json.loads((d/'failure-analysis.json').read_text())
        replay[name]=RecordedRetriever(reports[name],details[name],ps,qs)
    hm=json.loads((hybrid/'manifest.json').read_text())
    require(hm['corpus_fingerprint']==CORPUS and hm['evaluation_set_fingerprint']==fp,'Hybrid fingerprints')
    for dense,slug in [('E5','e5'),('BGE-M3','bge-m3')]:
        for variant in ['standard','gated']:
            name=f'{dense} original {variant} RRF';d=hybrid/f'{slug}-{variant}'
            reports[name]=json.loads((d/'evaluation.json').read_text())
            details[name]=json.loads((d/'failure-analysis.json').read_text())
        # Validate every replayed rank/provenance, including explicitly unanswerable queries.
        checker=RRFRetriever(replay['BM25'],replay[dense],require_complete=True);checker.index(ps)
        for q in qs:checker.search(q,15)
    out.mkdir(parents=True)
    runtimes={}
    models={'BM25 punctuation-filtered':PunctuationFilteredBM25Retriever(cfg.bm25)}
    for dense in ['E5','BGE-M3']:
        models[f'{dense} punctuation-filtered RRF']=RRFRetriever(
            PunctuationFilteredBM25Retriever(cfg.bm25),replay[dense],constant=60,
            gate_nonpositive_lexical=False,require_complete=True)
    for name,r in models.items():
        t=time.perf_counter();reports[name]=evaluate(r,qs,gold,docs,ps,cfg.evaluation);rows=[]
        for q in qs:
            hits=r.search(q,15);refs=byq[q.query_id]['evidence']
            def covered(items):
                return [ref['evidence_id'] for ref in refs if fully_covered(
                    EvidenceSpan.model_validate(ref['span']),[h.evidence_span for h in items])]
            missing={str(k):[ref['evidence_id'] for ref in refs if ref['evidence_id'] not in covered(hits[:k])] for k in spec['ks']}
            rows.append({'query_id':q.query_id,'group_id':byq[q.query_id]['group_id'],'language':q.language,
                'question':q.text,'answerable':labels[q.query_id].answerable,
                'required_evidence_ids':[ref['evidence_id'] for ref in refs],'missing_evidence_ids':missing,
                'all_required_covered':{str(k):not missing[str(k)] if labels[q.query_id].answerable else None for k in spec['ks']},
                'full_ranking':[{**h.model_dump(),'covered_gold_evidence_ids':covered([h])} for h in hits]})
        details[name]=rows;runtimes[name]=time.perf_counter()-t
        d=out/name.lower().replace(' ','-');d.mkdir();save(d/'evaluation.json',reports[name]);save(d/'failure-analysis.json',rows)
    for name,r in reports.items():
        require(r['schema_version']==2 and r['query_fingerprint']==reports['BM25']['query_fingerprint'] and
                r['label_fingerprint']==reports['BM25']['label_fingerprint'],'Report inputs mismatch')
    metrics=[]
    for lang in ['en','hi','hinglish']:
        for name,r in reports.items():metrics.append({'language':lang,'model':name,**r['per_language'][lang]})
    index={n:{row['query_id']:row for row in rows} for n,rows in details.items()}
    comparisons=[('BM25 punctuation-filtered','BM25'),('E5 punctuation-filtered RRF','E5 original standard RRF'),
                 ('BGE-M3 punctuation-filtered RRF','BGE-M3 original standard RRF')]
    changes=[]
    for new,old in comparisons:
        for q in qs:
            x=index[new][q.query_id];y=index[old][q.query_id]
            oldr={h['passage_id']:h['rank'] for h in y['full_ranking']}
            moves=[{'passage_id':h['passage_id'],'old_rank':oldr[h['passage_id']],'new_rank':h['rank']} for h in x['full_ranking'] if oldr[h['passage_id']]!=h['rank']]
            changes.append({'model':new,'reference':old,'query_id':q.query_id,'language':q.language,'answerable':x['answerable'],
                'rank_changes':moves,'old_missing':y['missing_evidence_ids'],'new_missing':x['missing_evidence_ids'],
                'old_complete':y['all_required_covered'],'new_complete':x['all_required_covered']})
    save(out/'comparison.json',{'scope':'Exploratory development analysis, 15 passages from one document; no tuning or new query approval.',
        'corpus_fingerprint':CORPUS,'evaluation_set_fingerprint':fp,'metrics':metrics,'rank_changes':changes,
        'diagnostics':{n:r['diagnostics'] for n,r in reports.items() if r['diagnostics']['available']},
        'runtime_seconds':runtimes})
    save(out/'all-ranked-results.json',details)
    require(all(sha(ROOT/p)==h for p,h in protected.items()),'Protected inputs changed')
    save(out/'manifest.json',{'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'corpus_fingerprint':CORPUS,'evaluation_set_fingerprint':fp,'annotation_sha256':sign['annotation_sha256'],
        'signoff_sha256':sha(a/'reviewer-signoff.json'),'configuration':spec,'configuration_sha256':sha(cfgpath),
        'protected_input_sha256':protected,'protected_inputs_unchanged':True,'script_sha256':sha(Path(__file__)),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').rglob('*.py')},
        'python':platform.python_version(),'packages':subprocess.check_output([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],text=True),
        'execution':'New lexical indexing; exact validated saved dense ranking replay. No encoder or generator loaded.'})
    save(out/'integrity.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps([{'language':r['language'],'model':r['model'],'complete':r['metrics']['complete_evidence_coverage_at_k']} for r in metrics],indent=2))


if __name__=='__main__':main()
