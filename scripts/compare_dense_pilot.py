"""Read-only comparison of completed fixed-corpus runs; writes a new local report."""
from krishisaarthi.validation import require
from pathlib import Path
import argparse,json,hashlib,html
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--e5',default='data/experiments/e5-pilot-v0.1-run-001');parser.add_argument('--bge',default='data/experiments/bge-m3-pilot-v0.1-run-001');parser.add_argument('--output',default='data/experiments/dense-comparison-v0.1-run-001');args=parser.parse_args()
    dirs={'BM25':ROOT/'data/experiments/bm25-pilot-v0.1-run-001','E5':ROOT/args.e5,'BGE-M3':ROOT/args.bge}
    out=(ROOT/args.output).resolve();require(out.is_relative_to(ROOT/'data') and not out.exists(), 'compare_dense_pilot.py:10: integrity validation failed')
    reports={};manifests={};detail={}
    for name,d in dirs.items():
        for file,h in json.loads((d/'integrity.json').read_text()).items():require(sha(d/file)==h, 'compare_dense_pilot.py:13: integrity validation failed')
        reports[name]=json.loads((d/'evaluation.json').read_text());manifests[name]=json.loads((d/'manifest.json').read_text())
        detail[name]={r['query_id']:r for r in json.loads((d/'failure-analysis.json').read_text())}
    for m in manifests.values():
        require(m['corpus_fingerprint']==manifests['BM25']['corpus_fingerprint'], 'compare_dense_pilot.py:17: integrity validation failed')
        require(m['evaluation_set_fingerprint']==manifests['BM25']['evaluation_set_fingerprint'], 'compare_dense_pilot.py:18: integrity validation failed')
    rows=[]
    for lang in ['en','hi','hinglish']:
        for model in dirs:
            v=reports[model]['per_language'][lang];mm=v['metrics']
            rows.append({'language':lang,'model':model,'scored':v['evaluated_queries'],'excluded':v['excluded_queries'],
                'recall':[mm[f'Recall@{k}'] for k in [1,3,5]],'cumulative_evidence_mrr':[mm['cumulative_evidence_mrr_at_k'][str(k)] for k in [1,3,5]],
                'complete_evidence_coverage':[mm['complete_evidence_coverage_at_k'][str(k)] for k in [1,3,5]]})
    recoveries={};hinglish_losses={}
    bmrows={r['query_id']:r for r in reports['BM25']['per_query']}
    for model in ['E5','BGE-M3']:
        ds={r['query_id']:r for r in reports[model]['per_query']}
        recoveries[model]={str(k):[q for q in bmrows if q.endswith('-hi') and not detail['BM25'][q]['all_required_covered'][str(k)] and detail[model][q]['all_required_covered'][str(k)]] for k in [1,3,5]}
        hinglish_losses[model]=[{ 'query_id':q,'k':k,'bm25':bmrows[q]['metrics'][str(k)],'dense':ds[q]['metrics'][str(k)]} for q in bmrows if q.endswith('-hinglish') for k in [1,3,5] if any(bmrows[q]['metrics'][str(k)][m]>ds[q]['metrics'][str(k)][m] for m in ['recall','reciprocal_rank','complete_evidence_coverage'])]
    disagreement=[];multispan=[];query_rows=[]
    for q,b in detail['BM25'].items():
        if detail['E5'][q]['full_ranking'][0]['passage_id']!=detail['BGE-M3'][q]['full_ranking'][0]['passage_id']:
            disagreement.append({'query_id':q,'answerable':b['answerable'],'e5_top1':detail['E5'][q]['full_ranking'][0]['passage_id'],'bge_top1':detail['BGE-M3'][q]['full_ranking'][0]['passage_id']})
        qr={'query_id':q,'group_id':b['group_id'],'question':b['question'],'answerable':b['answerable'],'models':{}}
        for model in dirs:
            d=detail[model][q]
            qr['models'][model]={'all_required_covered':d['all_required_covered'],'missing_evidence_ids':d['missing_evidence_ids'],
                'gold_ranks':{eid:next((r['rank'] for r in d['full_ranking'] if eid in r['covered_gold_evidence_ids']),None) for eid in d['required_evidence_ids']},
                'top5':d['full_ranking'][:5]}
        query_rows.append(qr)
        if b['group_id'][-2:] in ['08','10']:multispan.append(qr)
    token_runtime={}
    for model in ['E5','BGE-M3']:
        lengths=json.loads((dirs[model]/'token-lengths.json').read_text());m=manifests[model]
        token_runtime[model]={'device':m['inference']['device'],'batch_size':m['inference']['batch_size'],'dimension':m['model']['dimension'],'limit':m['model']['max_tokens'],'cache_hit':m['passage_cache_hit'],'timings':m['timings'],
            'passage_min_max':[min(v['tokens_with_prefix_and_specials'] for k,v in lengths.items() if k.startswith('passage:')),max(v['tokens_with_prefix_and_specials'] for k,v in lengths.items() if k.startswith('passage:'))],
            'query_min_max':[min(v['tokens_with_prefix_and_specials'] for k,v in lengths.items() if k.startswith('query:')),max(v['tokens_with_prefix_and_specials'] for k,v in lengths.items() if k.startswith('query:'))],
            'all_within_limit':all(v['within_limit'] for v in lengths.values())}
    comparison={'notice':'Exploratory 15-passage single-document pilot, no statistical significance claims. No tuning or corpus/label changes. Zero-score BM25 ties and punctuation matches are not semantic success; cosine scores are not lexical-match diagnostics.',
        'corpus_fingerprint':manifests['BM25']['corpus_fingerprint'],'evaluation_set_fingerprint':manifests['BM25']['evaluation_set_fingerprint'],
        'metrics':rows,'hindi_complete_evidence_recoveries_over_bm25':recoveries,'hinglish_where_bm25_outperforms':hinglish_losses,'e5_bge_top1_disagreements':disagreement,
        'multi_evidence_analysis':multispan,'tokens_and_runtime':token_runtime,'all_query_comparisons':query_rows,
        'input_files_sha256':{str(d.relative_to(ROOT))+'/'+f:sha(d/f) for d in dirs.values() for f in ['manifest.json','evaluation.json','failure-analysis.json']}}
    out.mkdir(parents=True)
    (out/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
    E=lambda x:html.escape(str(x))
    table=''.join('<tr>'+''.join('<td>'+E(x)+'</td>' for x in [r['language'],r['model'],' / '.join(f'{v:.4f}' for v in r['recall']),' / '.join(f'{v:.4f}' for v in r['cumulative_evidence_mrr']),' / '.join(f'{v:.4f}' for v in r['complete_evidence_coverage'])])+'</tr>' for r in rows)
    cards=[]
    for q in query_rows:
        body=''
        for model,d in q['models'].items():
            ranks=''.join('<tr>'+''.join('<td>'+E(x)+'</td>' for x in [r['rank'],r['passage_id'],f"{r['score']:.6f}",r['covered_gold_evidence_ids']])+'</tr>' for r in d['top5'])
            body+='<h3>'+model+'</h3><p>Gold ranks: '+E(d['gold_ranks'])+'<br>Missing @1/3/5: '+E(d['missing_evidence_ids'])+'<br>ALL covered: '+E(d['all_required_covered'])+'</p><details><summary>Top-5 scores, passage IDs and evidence IDs</summary><table><tr><th>Rank</th><th>Passage</th><th>Score</th><th>Evidence covered</th></tr>'+ranks+'</table></details>'
        cards.append('<section><h2>'+E(q['query_id'])+'</h2><p>'+E(q['question'])+'</p><p>Answerable: '+E(q['answerable'])+'</p>'+body+'</section>')
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BM25 vs E5 vs BGE-M3</title><style>body{font:15px system-ui;line-height:1.5;max-width:1350px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd5dd;padding:9px;text-align:left;overflow-wrap:anywhere}section{margin-top:32px;border-top:2px solid #ccd}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{overflow-wrap:anywhere}</style><h1>Dense-only pilot comparison</h1><p>'+E(comparison['notice'])+'</p><p>Per language: 12 answerable queries scored, 4 unanswerable excluded from evidence metrics. Values are K=1 / 3 / 5. All required spans count for complete coverage; cumulative MRR uses first covered span.</p><table><tr><th>Language</th><th>Model</th><th>Recall</th><th>Cumulative evidence MRR</th><th>Complete coverage</th></tr>'+table+'</table><h2>Hindi complete-evidence recoveries</h2><pre>'+E(json.dumps(recoveries,indent=2))+'</pre><h2>Hinglish cases where BM25 wins a metric</h2><pre>'+E(json.dumps(hinglish_losses,indent=2))+'</pre><h2>E5/BGE-M3 rank-1 disagreements</h2><pre>'+E(json.dumps(disagreement,indent=2))+'</pre><h2>Token counts and runtime</h2><pre>'+E(json.dumps(token_runtime,indent=2))+'</pre>'+''.join(cards)+'</html>')
    (out/'integrity.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps({k:comparison[k] for k in ['metrics','hindi_complete_evidence_recoveries_over_bm25','hinglish_where_bm25_outperforms','e5_bge_top1_disagreements','tokens_and_runtime']},indent=2))
if __name__=='__main__':main()
