"""Synthetic source intervals only; no private corpus or real model required."""
import copy
from pathlib import Path
import pytest
from krishisaarthi.schemas import Document,PageReference
from krishisaarthi.config import text_hash
from krishisaarthi.corpus.application import construct,build,load,POLICY


def fixture(tmp_path):
    text='SYNTHETIC barley seed observations. '+('A fictional sample was recorded. '*70)
    original=tmp_path/'synthetic.txt';original.write_text(text)
    d=Document(document_id='synthetic',content_sha256=text_hash(text),canonical_sha256=text_hash(text),original_filename='synthetic.txt',original_file_path=str(original),source_url=None,media_type='text/plain',language='en',domain_metadata={'crop':'barley'},canonical_text=text,pages=[PageReference(page_number=1,canonical_start=0,canonical_end=len(text),extracted_text_path=str(original))],extraction_metadata={})
    r={'region_id':'synthetic-region','crop':'barley','curation':'ai_assisted','human_review':'not_recorded','physical_page':1,'printed_page_label':'1','topic':'Synthetic observations','conditions':'Invented fixture only','limitations':'No agricultural facts','evidence_span':{'schema_version':1,'document_id':d.document_id,'content_sha256':d.content_sha256,'canonical_sha256':d.canonical_sha256,'start_char':0,'end_char':len(text)},'text_sha256':text_hash(text)}
    return d,{'policy':POLICY,'human_verified':False,'regions':[r],'excluded_spans':[]}


def test_deterministic_exact_provenance_and_chunking(tmp_path):
    d,plan=fixture(tmp_path);a=construct([d],[],plan);b=construct([d],[],plan)
    assert a==b and len(a)>1 and a[0].token_count==256
    for p in a:
        assert p.text==d.canonical_text[p.evidence_span.start_char:p.evidence_span.end_char]
        assert p.page_references==d.pages and p.domain_metadata['human_review']=='not_recorded'
    assert a[1].evidence_span.start_char<a[0].evidence_span.end_char


@pytest.mark.parametrize('bad',['content','source','page','span','held','human','crop','duplicate','page_file'])
def test_reject_invalid_selection(tmp_path,bad):
    d,p=fixture(tmp_path);r=p['regions'][0]
    if bad=='content':r['text_sha256']='0'*64
    elif bad=='source':r['evidence_span']['content_sha256']='0'*64
    elif bad=='page':r['physical_page']=2
    elif bad=='span':r['evidence_span']['end_char']+=1
    elif bad=='held':p['excluded_spans']=[{'document_id':'synthetic','start_char':0,'end_char':10}]
    elif bad=='human':p['human_verified']=True
    elif bad=='crop':r['crop']='unknown'
    elif bad=='duplicate':p['regions'].append(copy.deepcopy(r))
    else:Path(d.original_file_path).write_text('tampered')
    with pytest.raises(ValueError):construct([d],[],p)


def test_snapshot_roundtrip_and_tamper(tmp_path,monkeypatch):
    import krishisaarthi.corpus.application as module
    monkeypatch.setattr(module,'private_path',lambda p:p)
    d,p=fixture(tmp_path);out=tmp_path/'snapshot';m=build([d],[],p,out)
    docs,passages,loaded=load(out)
    assert loaded==m and docs==[d] and loaded['crop_counts']=={'barley':len(passages)}
    with pytest.raises(ValueError):build([d],[],p,out)
    (out/'passages.jsonl').write_text('changed')
    with pytest.raises(ValueError):load(out)


def test_crop_filter_preserves_ranking_and_exact_provenance(tmp_path):
    import threading
    from krishisaarthi.schemas import RetrievalResult
    from krishisaarthi.ui.local import LocalRetrieval
    d,plan=fixture(tmp_path);passages=construct([d],[],plan)
    # A higher-ranked different crop must not leak into the selected crop.
    passages[0].domain_metadata['crop']='wheat'
    class Retriever:
        _query_vectors={}
        token_lengths={}
        def search(self,q,k):
            return [RetrievalResult(query_id=q.query_id,passage_id=p.passage_id,document_id=p.document_id,rank=i+1,score=1/(i+1),evidence_span=p.evidence_span,retriever_id='synthetic') for i,p in enumerate(passages[:k])]
    service=LocalRetrieval.__new__(LocalRetrieval)
    service.documents={d.document_id:d};service.passages={p.passage_id:p for p in passages}
    service.retriever=Retriever();service.lock=threading.Lock();service.corpus_fingerprint='synthetic-only'
    service.collection_info={'version':'synthetic','passage_count':len(passages),'crop_counts':{'wheat':1,'barley':len(passages)-1}}
    for language in ('en','hi','hinglish'):
        result=service.search({'query':'SYNTHETIC','language':language,'crop':'barley'})
        assert result['results'][0]['rank']==1
        assert result['results'][0]['passage_id']==passages[1].passage_id
        assert all(r['metadata']['crop']=='barley' for r in result['results'])
        assert result['corpus_fingerprint']=='synthetic-only'
    with pytest.raises(ValueError):service.search({'query':'x','language':'en','crop':'chickpea'})


def test_application_context_encoding_is_separate_from_research(tmp_path):
    from test_dense import FakeEncoder,SPECS
    from krishisaarthi.retrieval.application_e5 import ApplicationE5
    from krishisaarthi.retrieval.e5 import E5Retriever
    from krishisaarthi.ui.local import ApplicationCachedE5
    d,plan=fixture(tmp_path);passages=construct([d],[],plan)
    enc=FakeEncoder(768);app=ApplicationE5(enc,SPECS['e5'],'synthetic',tmp_path/'cache')
    before=[p.model_dump() for p in passages];app.index(passages)
    formatted=app.formatted([passages[0].text],'passage')[0]
    assert formatted.startswith('passage: Crop: Barley.') and formatted.endswith(passages[0].text)
    assert app.formatted(['synthetic query'],'query')==['query: synthetic query']
    assert [p.model_dump() for p in passages]==before
    plain=E5Retriever(FakeEncoder(768),SPECS['e5'],'synthetic')
    assert plain.model_fingerprint!=app.model_fingerprint
    reader=ApplicationCachedE5(FakeEncoder(768),SPECS['e5'],'synthetic',tmp_path/'cache')
    reader.index(passages)
    assert reader.cache_hit and not reader.encoder.seen
    with pytest.raises(FileNotFoundError):
        ApplicationCachedE5(FakeEncoder(768),SPECS['e5'],'wrong-corpus',tmp_path/'cache').index(passages)


def test_application_context_is_included_in_token_limit(tmp_path):
    from test_dense import FakeEncoder,SPECS
    from krishisaarthi.retrieval.application_e5 import ApplicationE5
    from krishisaarthi.retrieval.dense import TokenLimitError
    d,plan=fixture(tmp_path);passages=construct([d],[],plan)
    class BoundaryEncoder(FakeEncoder):
        def token_counts(self,texts):
            return [513 if 'Crop:' in text else 511 for text in texts]
    enc=BoundaryEncoder(768)
    with pytest.raises(TokenLimitError):ApplicationE5(enc,SPECS['e5'],'synthetic').index(passages)
    assert not enc.seen


def test_source_qualifications_are_exact_separate_citations(tmp_path):
    d,plan=fixture(tmp_path);r=plan['regions'][0]
    r['source_qualifications']=[{'evidence_span':copy.deepcopy(r['evidence_span']),
        'physical_page':1,'text_sha256':r['text_sha256']}]
    passages=construct([d],[],plan)
    q=passages[0].domain_metadata['source_qualifications'][0]
    assert q['text']==d.canonical_text and q['filename']=='synthetic.txt'
    assert q['evidence_span']==r['evidence_span']
    assert passages[0].text==d.canonical_text[:passages[0].evidence_span.end_char]


@pytest.mark.parametrize('bad',['hash','page','offset','source'])
def test_reject_invalid_source_qualification(tmp_path,bad):
    d,plan=fixture(tmp_path);r=plan['regions'][0]
    q={'evidence_span':copy.deepcopy(r['evidence_span']),'physical_page':1,'text_sha256':r['text_sha256']}
    r['source_qualifications']=[q]
    if bad=='hash':q['text_sha256']='0'*64
    elif bad=='page':q['physical_page']=2
    elif bad=='offset':q['evidence_span']['end_char']+=1
    else:q['evidence_span']['content_sha256']='0'*64
    with pytest.raises(ValueError):construct([d],[],plan)
