"""Synthetic benchmark records only; fictional review identity is never human signoff."""
import hashlib
import json
from pathlib import Path
import pytest
from krishisaarthi.config import Config, EvaluationConfig, fingerprint
from krishisaarthi.schemas import SourceMetadata, EvidenceSpan
from krishisaarthi.ingestion.storage import make_document, save_document, write_jsonl
from krishisaarthi.corpus.models import ReviewedRegion, ReviewChecks
from krishisaarthi.corpus.review import build_passages, review_fingerprint
from krishisaarthi.evaluation.benchmark import (BenchmarkConfig, Dataset, load_snapshot,
    validate_dataset, run_benchmark, group_macro)
from krishisaarthi.retrieval.bm25 import BM25Retriever
from krishisaarthi.evaluation.runner import aggregate_metrics


def dump(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False))
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def benchmark(tmp_path):
    source = tmp_path/'source';source.mkdir()
    original = tmp_path/'fictional.txt';original.write_text('SYNTHETIC ONLY\n'+'seedmarker '*300+'watermarker '*300)
    d = save_document(make_document(original, SourceMetadata(document_id='fictional'),
        [(1, original.read_text())], 'text/plain', 'synthetic', '1'), source)
    span = EvidenceSpan(document_id=d.document_id, content_sha256=d.content_sha256,
        canonical_sha256=d.canonical_sha256, start_char=0, end_char=len(d.canonical_text))
    r = ReviewedRegion(region_id='fictional-region',evidence_span=span,physical_page=1,recommendation='candidate')
    r.checks = ReviewChecks(**{k:True for k in ReviewChecks.model_fields if k!='schema_version'})
    r.reviewer='SYNTHETIC_NOT_A_HUMAN';r.reviewed_at='2000-01-01T00:00:00+00:00';r.review_note='Synthetic test only'
    r.reviewed_content_fingerprint=review_fingerprint(r);r.status='approved'
    write_jsonl(source/'documents.jsonl',[d]);write_jsonl(source/'regions.jsonl',[r]);write_jsonl(source/'issues.jsonl',[])
    (source/'registry.jsonl').write_text('{}\n')
    sf=fingerprint(sorted((x.document_id,x.content_sha256,x.canonical_sha256) for x in [d]))
    dump(source/'integrity.json',{'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.jsonl') if p.name!='regions.jsonl'},'corpus_fingerprint':sf})
    cfg=Config(evaluation=EvaluationConfig(ks=[1,3,5]));passages,selection=build_passages([d],[r],[],cfg.chunking)
    snap=tmp_path/'snapshot';snap.mkdir();write_jsonl(snap/'approved-passages.jsonl',passages);write_jsonl(snap/'approved-regions.jsonl',[r])
    fp=fingerprint([p.model_dump() for p in passages]);mh=dump(snap/'index-manifest.json',{'passages_fingerprint':fp,'selection_fingerprint':selection,'source_corpus_fingerprint':sf,'chunking':cfg.chunking.model_dump(),'passages':len(passages),'approved_regions':1})
    sign={'decision':'accept','reviewer_id':'SYNTHETIC_NOT_A_HUMAN','reviewed_at':'2000-01-01T00:00:00+00:00','note':'Synthetic test only'}
    data={'corpus_version':'synthetic-v1','corpus_fingerprint':fp,'groups':[{'group_id':'g','split':'development','evidence_family_ids':['f'],'split_block_ids':['block'],'answerable':True,'required_evidence_ids':['e1','e2'],'signoff':sign}],
        'queries':[{'query_id':lang,'text':'seedmarker','language':lang,'group_id':'g','signoff':sign} for lang in ['en','hi','hinglish']],
        'evidence':[{'evidence_id':'e1','span':passages[0].evidence_span.model_dump()},{'evidence_id':'e2','span':passages[-1].evidence_span.model_dump()}]}
    ah=dump(tmp_path/'annotations.json',data)
    c=BenchmarkConfig(corpus_version='synthetic-v1',source_bundle='source',approved_snapshot='snapshot',source_fingerprint=sf,corpus_fingerprint=fp,approved_regions_sha256=hashlib.sha256((snap/'approved-regions.jsonl').read_bytes()).hexdigest(),index_manifest_sha256=mh,annotations='annotations.json',annotations_sha256=ah)
    return tmp_path,c,data,d,passages


def test_deterministic_multispan_run(benchmark):
    base,c,data,d,ps=benchmark
    a=run_benchmark(c,base,BM25Retriever());b=run_benchmark(c,base,BM25Retriever())
    assert a==b
    assert a['benchmark']['group_macro']['complete_evidence_coverage_at_k']=={'1':0.,'3':1.,'5':1.}
    assert a['metrics']['cumulative_evidence_mrr_at_k']['3']>0
    assert a['schema_version']==2


def test_multiple_corpus_versions(benchmark):
    base,c,data,d,ps=benchmark
    data['corpus_version']='synthetic-v2'
    c=c.model_copy(update={'corpus_version':'synthetic-v2','annotations_sha256':dump(base/'annotations.json',data)})
    assert run_benchmark(c,base,BM25Retriever())['benchmark']['corpus_version']=='synthetic-v2'


@pytest.mark.parametrize('kind',['fingerprint','text','page','pending','zero','manifest','source'])
def test_snapshot_fail_closed(benchmark,kind):
    base,c,data,d,ps=benchmark;snap=base/'snapshot'
    if kind=='fingerprint':c=c.model_copy(update={'corpus_fingerprint':'0'*64})
    elif kind in ['text','page']:
        ps[0].text='tampered' if kind=='text' else ps[0].text
        if kind=='page':ps[0].page_references[0].page_number=2
        write_jsonl(snap/'approved-passages.jsonl',ps)
    elif kind in ['pending','zero']:
        p=snap/'approved-regions.jsonl'
        if kind=='zero':p.write_text('')
        else:
            r=json.loads(p.read_text());r['status']='pending';p.write_text(json.dumps(r)+'\n')
        c=c.model_copy(update={'approved_regions_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    elif kind=='manifest':(snap/'index-manifest.json').write_text('{}')
    else:Path(d.original_file_path).write_text('tampered')
    with pytest.raises(ValueError):load_snapshot(c,base)


@pytest.mark.parametrize('kind',['family','block','overlap','duplicate_query','duplicate_group','duplicate_evidence','missing_evidence','missing_language','missing_group','pending','empty_split','version','uncoverable'])
def test_annotation_validation(benchmark,kind):
    base,c,data,d,ps=benchmark
    if kind in ['family','block','overlap']:
        g={**data['groups'][0],'group_id':'other','split':'test','evidence_family_ids':['f' if kind=='family' else 'f2'],'split_block_ids':['block' if kind=='block' else 'other-block']};data['groups'].append(g)
        data['queries'] += [{**q,'query_id':'other-'+q['query_id'],'group_id':'other'} for q in list(data['queries'])]
    elif kind=='duplicate_query':data['queries'].append(data['queries'][0])
    elif kind=='duplicate_group':data['groups'].append(data['groups'][0])
    elif kind=='duplicate_evidence':data['evidence'].append(data['evidence'][0])
    elif kind=='missing_evidence':data['groups'][0]['required_evidence_ids']=['missing']
    elif kind=='missing_language':data['queries'].pop()
    elif kind=='missing_group':data['queries'][0]['group_id']='unknown'
    elif kind=='pending':data['queries'][0]['signoff']={**data['queries'][0]['signoff'],'decision':'pending'}
    elif kind=='empty_split':c=c.model_copy(update={'split':'test'})
    elif kind=='version':data['corpus_version']='wrong'
    elif kind=='uncoverable':data['evidence'][0]['span']['end_char']=len(d.canonical_text)+1
    with pytest.raises(ValueError):validate_dataset(Dataset.model_validate(data),c,[d],ps)


def test_unanswerable_only_kept_but_not_scored(benchmark):
    base,c,data,d,ps=benchmark
    data['groups'][0].update(answerable=False,required_evidence_ids=[]);data['evidence']=[]
    c=c.model_copy(update={'annotations_sha256':dump(base/'annotations.json',data)})
    result=run_benchmark(c,base,BM25Retriever())
    assert result['excluded_queries']==3
    assert result['benchmark']['group_macro']['group_macro_complete_evidence_coverage_at_3'] is None


def test_annotation_hash_rejected(benchmark):
    base,c,*_=benchmark;(base/'annotations.json').write_text('{}')
    with pytest.raises(ValueError,match='fingerprint'):run_benchmark(c,base,BM25Retriever())


def test_group_macro_excludes_stress_weight(benchmark):
    _,_,data,_,_=benchmark
    data['queries'].append({**data['queries'][0],'query_id':'stress','variant_type':'stress'})
    ds=Dataset.model_validate(data)
    rows=[{'query_id':q.query_id,'language':q.language,'relevance_type':'evidence_spans',
        'metrics':{str(k):{'recall':float(q.variant_type=='base'),'reciprocal_rank':float(q.variant_type=='base'),'complete_evidence_coverage':float(q.variant_type=='base')} for k in [1,3,5]}} for q in ds.queries]
    assert aggregate_metrics(rows,[1,3,5])['complete_evidence_coverage_at_k']['3']==.75
    assert group_macro(rows,ds.queries,ds.groups)['group_macro_complete_evidence_coverage_at_3']==1
    with pytest.raises(ValueError):group_macro(rows[1:],ds.queries,ds.groups)


def test_protocol_configuration_rejects_changes(benchmark):
    _,c,*_=benchmark
    raw=c.model_dump();raw['settings']['chunking']['size']=128
    with pytest.raises(ValueError):BenchmarkConfig.model_validate(raw)


def test_resigned_tampered_passage_still_rejected(benchmark):
    base,c,_,_,ps=benchmark;snap=base/'snapshot'
    ps[0].text='synthetic tampered content'
    write_jsonl(snap/'approved-passages.jsonl',ps)
    fp=fingerprint([p.model_dump() for p in ps]);m=json.loads((snap/'index-manifest.json').read_text());m['passages_fingerprint']=fp
    c=c.model_copy(update={'corpus_fingerprint':fp,'index_manifest_sha256':dump(snap/'index-manifest.json',m)})
    with pytest.raises(ValueError,match='provenance'):load_snapshot(c,base)


def test_empty_passage_snapshot_rejected(benchmark):
    base,c,*_=benchmark;(base/'snapshot/approved-passages.jsonl').write_text('')
    with pytest.raises(ValueError):load_snapshot(c,base)
