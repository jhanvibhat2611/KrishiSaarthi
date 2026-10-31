"""Frozen-input RRF experiment: replay validated complete baseline rankings."""
import argparse
import hashlib
import html
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from krishisaarthi.config import fingerprint, load_config
from krishisaarthi.schemas import Document, Passage, Query, GoldLabel, RetrievalResult, EvidenceSpan
from krishisaarthi.ingestion.storage import read_jsonl
from krishisaarthi.evaluation.labels import validate_labels
from krishisaarthi.evaluation.metrics import fully_covered
from krishisaarthi.evaluation.runner import evaluate
from krishisaarthi.retrieval.rrf import RRFRetriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS = 'a919560be507b862be119a236fd1d2501f8d7c32d8fe6ea4951114b205f8c88a'

def require(condition, message):
    if not condition: raise ValueError(message)

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def ledger(d):
    for name, expected in json.loads((d/'integrity.json').read_text()).items():
        p=(d/name).resolve()
        require(p.is_relative_to(d.resolve()) and sha(p)==expected, f'Integrity: {p}')

class RecordedRetriever:
    """Exact offline replay; rejects new queries, changed passages or incomplete lists."""
    def __init__(self, report, details, passages, queries):
        self.retriever_id=report['retriever_id']
        self.corpus=fingerprint([p.model_dump() for p in passages])
        self.queries={q.query_id:q for q in queries}
        self.rankings={}
        fields=set(RetrievalResult.model_fields)
        for row in details:
            qid=row['query_id']
            require(qid not in self.rankings and qid in self.queries, 'Duplicate/unknown query')
            require(row['question']==self.queries[qid].text, 'Query text mismatch')
            hits=[RetrievalResult.model_validate({k:v for k,v in r.items() if k in fields}) for r in row['full_ranking']]
            require(len(hits)==len(passages), 'Incomplete saved ranking')
            require([r.passage_id for r in hits]==[r.passage_id for r in sorted(hits,key=lambda r:(-r.score,r.passage_id))], 'Input score order mismatch')
            self.rankings[qid]=hits
        require(self.rankings.keys()==self.queries.keys(),'Missing query')
    def index(self, passages):
        require(fingerprint([p.model_dump() for p in passages])==self.corpus, 'Changed corpus')
    def search(self, query, k):
        require(query==self.queries.get(query.query_id), 'Changed query')
        return self.rankings[query.query_id][:k]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='data/experiments/hybrid-comparison-v0.1-run-001')
    parser.add_argument('--constant',type=float,default=60)
    args=parser.parse_args(); started=datetime.now(timezone.utc).isoformat()
    out=(ROOT/args.output).resolve()
    require(out.is_relative_to(ROOT/'data') and not out.exists(),'Use fresh local data output')
    c=ROOT/'data/pilot-v0.1'; a=ROOT/'data/annotation-final-signoff-v1'
    dirs={name:ROOT/f'data/experiments/{slug}-pilot-v0.1-run-001' for name,slug in [('BM25','bm25'),('E5','e5'),('BGE-M3','bge-m3')]}
    for d in [c,*dirs.values()]:ledger(d)
    protected={str(p):sha(p) for d in [c,a,*dirs.values()] for p in d.rglob('*') if p.is_file()}
    docs=read_jsonl(c/'documents.jsonl',Document); passages=read_jsonl(c/'passages.jsonl',Passage)
    b=dirs['BM25'];queries=read_jsonl(b/'queries.jsonl',Query);gold=read_jsonl(b/'gold.jsonl',GoldLabel)
    metadata=json.loads((b/'query-evidence-metadata.json').read_text()); byid={r['query_id']:r for r in metadata}
    require(len(passages)==15 and len(queries)==48 and len(gold)==48 and sum(g.answerable for g in gold)==36,'Counts')
    require(len({r['group_id'] for r in metadata})==16 and byid.keys()=={q.query_id for q in queries},'Group metadata')
    require(not validate_labels(queries,gold,docs,passages)[2], 'Label validation warnings')
    labels={g.query_id:g for g in gold}
    for q in queries:
        if byid[q.query_id]['group_id'][-2:] in ['08','10']:
            require(byid[q.query_id]['evidence_requirement']=='ALL' and len(labels[q.query_id].evidence_spans)==2, 'ALL requirement')
    fp=fingerprint({'queries':[q.model_dump() for q in queries],'gold':[g.model_dump() for g in gold],'metadata':metadata})
    sign=json.loads((a/'reviewer-signoff.json').read_text())
    require(sign['status']=='signed_off' and sign['final_decision']=='accept','Signoff incomplete')
    cfg=load_config(b/'config.toml'); require(cfg.evaluation.ks==[1,3,5],'K configuration')
    reports={};details={};inputs={}; manifests={}
    for name,d in dirs.items():
        m=json.loads((d/'manifest.json').read_text());manifests[name]=m
        require(m['corpus_fingerprint']==CORPUS and m['evaluation_set_fingerprint']==fp,'Fingerprint')
        require(m['annotation_sha256']==sha(a/'annotation-candidate.json')==sign['annotation_sha256'] and m['signoff_sha256']==sha(a/'reviewer-signoff.json'),'Signed annotation')
        reports[name]=json.loads((d/'evaluation.json').read_text())
        details[name]=json.loads((d/'failure-analysis.json').read_text())
        inputs[name]=RecordedRetriever(reports[name],details[name],passages,queries)
    # Validate every saved row including unanswerable queries through RRF's provenance checks.
    for dense in ['E5','BGE-M3']:
        check=RRFRetriever(inputs['BM25'],inputs[dense],require_complete=True);check.index(passages)
        for q in queries:check.search(q,15)
    out.mkdir(parents=True)
    def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    runtimes={}; configs={}
    for dense in ['E5','BGE-M3']:
        for gated in [False,True]:
            name=f'BM25 + {dense} '+('gated diagnostic' if gated else 'standard RRF')
            retriever=RRFRetriever(inputs['BM25'],inputs[dense],constant=args.constant,
                gate_nonpositive_lexical=gated,require_complete=True)
            t=time.perf_counter();report=evaluate(retriever,queries,gold,docs,passages,cfg.evaluation)
            rows=[]
            for q in queries:
                hits=retriever.search(q,15); refs=byid[q.query_id]['evidence']; label=labels[q.query_id]
                def covered(items):return [r['evidence_id'] for r in refs if fully_covered(EvidenceSpan.model_validate(r['span']),[h.evidence_span for h in items])]
                missing={str(k):[r['evidence_id'] for r in refs if r['evidence_id'] not in covered(hits[:k])] for k in [1,3,5]}
                rows.append({'query_id':q.query_id,'group_id':byid[q.query_id]['group_id'],'language':q.language,'question':q.text,
                    'answerable':label.answerable,'required_evidence_ids':[r['evidence_id'] for r in refs],
                    'missing_evidence_ids':missing,'all_required_covered':{str(k):not missing[str(k)] if label.answerable else None for k in [1,3,5]},
                    'full_ranking':[{**h.model_dump(),'covered_gold_evidence_ids':covered([h])} for h in hits]})
            runtimes[name]=time.perf_counter()-t;reports[name]=report;details[name]=rows
            slug=('e5' if dense=='E5' else 'bge-m3')+('-gated' if gated else '-standard');d=out/slug;d.mkdir()
            config={'constant':args.constant,'weights':[1,1],'dense':dense,'lexical':'BM25','gate_bm25_score_le_zero':gated,
                'gated_rank_policy':'original rank, no compaction','input_depth':15,'ks':[1,3,5],'tie_break':'passage_id',
                'execution':'exact replay of validated saved full baseline rankings; no model re-encoding'}
            configs[name]=config;save(d/'config.json',config);save(d/'evaluation.json',report);save(d/'failure-analysis.json',rows)
    metric_rows=[]
    for lang in ['en','hi','hinglish']:
        for name,r in reports.items():
            m=r['per_language'][lang]['metrics']
            metric_rows.append({'language':lang,'model':name,'recall':[m[f'Recall@{k}'] for k in [1,3,5]],
                'cumulative_evidence_mrr':[m['cumulative_evidence_mrr_at_k'][str(k)] for k in [1,3,5]],
                'complete_evidence_coverage':[m['complete_evidence_coverage_at_k'][str(k)] for k in [1,3,5]]})
    metrics={name:{r['query_id']:r['metrics'] for r in report['per_query']} for name,report in reports.items()}
    changes=[]
    for name,conf in configs.items():
        for qid,mm in metrics[name].items():
            for k in ['1','3','5']:
                for metric,value in mm[k].items():
                    left=metrics['BM25'][qid][k][metric];right=metrics[conf['dense']][qid][k][metric]
                    if value!=left or value!=right:
                        changes.append({'hybrid':name,'query_id':qid,'k':int(k),'metric':metric,'hybrid_value':value,'bm25':left,'dense':right,
                            'better_than_both':value>max(left,right),'worse_than_both':value<min(left,right)})
    indexed={n:{r['query_id']:r for r in rows} for n,rows in details.items()}
    perquery=[]
    for q in queries:
        perquery.append({'query_id':q.query_id,'language':q.language,'question':q.text,'answerable':labels[q.query_id].answerable,
            'models':{name:{'top5':data[q.query_id]['full_ranking'][:5],
                'gold_ranks':{eid:next((h['rank'] for h in data[q.query_id]['full_ranking'] if eid in h['covered_gold_evidence_ids']),None) for eid in data[q.query_id]['required_evidence_ids']},
                'missing_evidence_ids':data[q.query_id]['missing_evidence_ids']} for name,data in indexed.items()}})
    comparison={'notice':'Exploratory 15-passage single-document pilot. Equal-weight RRF; gated variant is a diagnostic, not a tuned or novel method.',
        'corpus_fingerprint':CORPUS,'evaluation_set_fingerprint':fp,'metrics':metric_rows,'paired_metric_changes':changes,
        'per_query':perquery,'multi_span_queries':[r for r in perquery if '-in-08-' in r['query_id'] or '-in-10-' in r['query_id']],
        'bm25_diagnostics':reports['BM25']['diagnostics'],'configs':configs,'runtime_seconds_replay_evaluation_and_analysis':runtimes}
    save(out/'comparison.json',comparison)
    require(all(sha(Path(p))==h for p,h in protected.items()),'Protected inputs changed')
    save(out/'manifest.json',{'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'corpus_fingerprint':CORPUS,
        'evaluation_set_fingerprint':fp,'annotation_sha256':sign['annotation_sha256'],'signoff_sha256':sha(a/'reviewer-signoff.json'),
        'protected_inputs_sha256':protected,'input_run_manifests':manifests,'configs':configs,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').rglob('*.py')},
        'script_sha256':sha(Path(__file__)),'python':platform.python_version(),'packages':subprocess.check_output([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],text=True),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protected_inputs_unchanged':True})
    E=lambda v:html.escape(str(v))
    table=''.join('<tr>'+''.join('<td>'+E(x)+'</td>' for x in [r['language'],r['model'],*[' / '.join(f'{v:.4f}' for v in r[m]) for m in ['recall','cumulative_evidence_mrr','complete_evidence_coverage']]])+'</tr>' for r in metric_rows)
    cards=[]
    for row in perquery:
        body=''
        for name,values in row['models'].items():
            body+='<h3>'+E(name)+'</h3><p>Evidence ranks: '+E(values['gold_ranks'])+'<br>Missing @1/3/5: '+E(values['missing_evidence_ids'])+'</p><details><summary>Top-5 passages, scores and evidence</summary><pre>'+E(json.dumps(values['top5'],indent=2))+'</pre></details>'
        cards.append('<details><summary>'+E(row['query_id']+' — '+row['question'])+'</summary>'+body+'</details>')
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Phase 3A RRF comparison</title><style>body{font:15px system-ui;max-width:1450px;margin:25px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:16px 0}</style><h1>BM25, dense and RRF pilot comparison</h1><p>'+E(comparison['notice'])+'</p><p>K = 1 / 3 / 5. Per language: 12 answerable scored; 4 unanswerable excluded. Cumulative MRR requires any span; complete coverage requires ALL. Standard RRF includes zero-score ties; diagnostic gating excludes BM25 scores ≤0. Dense/RRF scores are not lexical match probabilities.</p><table><tr><th>Language</th><th>Model</th><th>Recall</th><th>Cumulative evidence MRR</th><th>Complete coverage</th></tr>'+table+'</table><p><a href="robustness-proposal.txt">Proposed code-mixing experiment</a> · <a href="validation-summary.txt">Validation and Git summary</a></p><h2>Strict gains/losses against both inputs</h2><pre>'+E(json.dumps([r for r in changes if r['better_than_both'] or r['worse_than_both']],indent=2))+'</pre><h2>All 48 queries</h2>'+''.join(cards)+'</html>')
    save(out/'integrity.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(metric_rows,indent=2))

if __name__=='__main__':main()
