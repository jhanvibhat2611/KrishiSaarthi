"""Glossary tests use questions only, no copyrighted passages or gold labels."""
import pytest
from krishisaarthi.ui.hinglish import normalize
from krishisaarthi.ui.local import LocalRetrieval, ApplicationRetrieval
from krishisaarthi.ui.server import validate_payload

@pytest.mark.parametrize('q,words',[
 ('gehu kab ugana chahiye',['wheat','when','grow','should']),
 ('GEHUN ki buwai kab kare?',['wheat','sowing','when']),
 ('makka ko paani kab dena chahiye?',['maize','water','when']),
 ('chana kis mitti mein ugaye?',['chickpea','soil','grow']),
 ('sarso ki katai kab kare?',['mustard','harvesting','when']),
 ('barley ke liye kitna beej chahiye?',['barley','seed','how much']),
 ('gehu mein urea kab nahi dena chahiye',['wheat','urea','when','not']),
 ('fasal 50% se kam aur barish pehle',['crop','50%','less than','and','before']),
])
def test_general_vocabulary(q,words):
    r=normalize(q,'hinglish')
    assert r['original_query']==q and r['applied']
    assert all(x in r['search_query'] for x in words)
    assert r['replacements']

@pytest.mark.parametrize('language,enabled',[('en',True),('hi',True),('hinglish',False)])
def test_unchanged_without_opt_in(language,enabled):
    q='gehu kab ugana chahiye'
    assert normalize(q,language,enabled)['search_query']==q


def test_boundaries_identifiers_quantities_and_unknown_words():
    q='PBW 869 DBW-187 20 kg/acre 10% glyphosatexyz gehuvirus jau nahi aur ya'
    r=normalize(q,'hinglish')['search_query']
    assert r=='PBW 869 DBW-187 20 kg/acre 10% glyphosatexyz gehuvirus jau not and or'
    assert normalize('गेहूँ की बुवाई mystery123','hinglish')['search_query']=='गेहूँ की बुवाई mystery123'


def test_application_only_and_boolean_validation():
    assert LocalRetrieval.normalize_hinglish is False
    assert ApplicationRetrieval.normalize_hinglish is True
    for bad in ['false',0,None,[],{}]:
        with pytest.raises(ValueError):validate_payload({'query':'x','language':'hinglish','normalize_hinglish':bad})


def test_search_adapter_sends_only_transformed_question_and_retains_original(tmp_path):
    import threading
    from test_application_collection import fixture
    from krishisaarthi.corpus.application import construct
    from krishisaarthi.schemas import RetrievalResult
    d,plan=fixture(tmp_path);ps=construct([d],[],plan);seen=[]
    class Retriever:
        _query_vectors={}
        token_lengths={}
        def search(self,q,k):
            seen.append(q.text)
            return [RetrievalResult(query_id=q.query_id,passage_id=p.passage_id,document_id=p.document_id,rank=i+1,score=1/(i+1),evidence_span=p.evidence_span,retriever_id='synthetic') for i,p in enumerate(ps[:k])]
    service=ApplicationRetrieval.__new__(ApplicationRetrieval)
    service.documents={d.document_id:d};service.passages={p.passage_id:p for p in ps}
    service.retriever=Retriever();service.lock=threading.Lock();service.corpus_fingerprint='synthetic'
    service.collection_info={'crop_counts':{'barley':len(ps)}}
    before=[p.model_dump() for p in ps]
    q='barley ki katai kab kare'
    r=service.search({'query':q,'language':'hinglish','crop':'barley'})
    assert seen==[r['query_transformation']['search_query']] and seen[0]!=q
    assert r['query_transformation']['original_query']==q
    assert [p.model_dump() for p in ps]==before
    service.search({'query':q,'language':'hinglish','normalize_hinglish':False})
    assert seen[-1]==q


def test_jau_requires_explicit_barley_context():
    assert normalize('jau', 'hinglish')['search_query']=='jau'
    assert normalize('jau', 'hinglish', crop='wheat')['search_query']=='jau'
    assert normalize('jau', 'hinglish', crop='barley')['search_query']=='barley'
    assert normalize('jau', 'hinglish', False, crop='barley')['search_query']=='jau'


def test_spacing_and_rain_preserve_logical_conditions():
    q='line se line doori 42 cm nahi aur baarish ke bharose dobara bone'
    r=normalize(q,'hinglish')
    assert r['original_query']==q
    assert r['search_query']=='row to row spacing 42 cm not and rainfed re-sowing'
