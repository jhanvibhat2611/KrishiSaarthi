"""Synthetic-only UI boundary and HTTP contract tests."""
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from krishisaarthi.ui.server import search, Handler, ThreadingHTTPServer

@pytest.mark.parametrize('query,language', [('soil','en'),('मिट्टी','hi'),('mitti','hinglish')])
def test_languages_and_synthetic_citations(query, language):
    result = search({'query':query,'language':language})
    assert result['mode']=='synthetic' and result['live_available'] is False
    assert len(result['results']) == 1
    for item in result['results']:
        assert item['text'].startswith('SYNTHETIC:')
        assert item['citation'].startswith('DEMO ·')
        assert item['passage_id'].startswith('synthetic-')
        assert 'score' not in item and 'page' not in item

@pytest.mark.parametrize('payload',[{},None,[],{'query':' ','language':'en'},{'query':'x'*1001,'language':'en'},{'query':'soil','language':'xx'}])
def test_validation(payload):
    with pytest.raises(ValueError): search(payload)

def test_empty_and_deterministic():
    assert search({'query':'unmatchedxyz','language':'en'})['results']==[]
    q={'query':'wheat','language':'en'}
    assert search(q)==search(q)
    assert len(search(q)['results'])==3

@pytest.fixture
def server():
    s=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
    yield f'http://127.0.0.1:{s.server_port}'
    s.shutdown();s.server_close();t.join()

def test_static_allowlist_and_no_research_routes(server):
    with urlopen(server) as r:
        assert b'KrishiSaarthi' in r.read()
    with urlopen(server + '/field-hero.jpg') as r:
        assert r.headers['Content-Type'].startswith('image/jpeg')
        assert r.read().startswith(b'\xff\xd8')
        assert "default-src 'self'" in r.headers['Content-Security-Policy']
    for path in ['/data/pilot-v0.1/passages.jsonl','/../pyproject.toml','/api/live','/source.html']:
        with pytest.raises(HTTPError) as e: urlopen(server+path)
        assert e.value.code==404

def test_http_search_and_errors(server):
    body=json.dumps({'query':'soil','language':'en'}).encode()
    with urlopen(Request(server+'/api/search',data=body)) as r:
        assert json.load(r)['results'][0]['document_id']=='demo-notebook-soil'
    for body in [b'bad',b'{}',b'[]',b'x'*8193]:
        with pytest.raises(HTTPError) as e: urlopen(Request(server+'/api/search',data=body))
        assert e.value.code==400
    with pytest.raises(HTTPError) as e:
        urlopen(Request(server+'/api/search',data=b'{}',headers={'Origin':'https://example.invalid'}))
    assert e.value.code==403


def test_qa_http_contract(server):
    from krishisaarthi.ui.qa import QUESTIONS
    for language in ['en','hi','hinglish']:
        payload={'query':QUESTIONS['soil'][language],'language':language}
        with urlopen(Request(server+'/api/answer',data=json.dumps(payload).encode())) as r:
            data=json.load(r)
            assert data['qa']['status']=='review_required'
            assert data['qa']['language']==language
            assert data['qa']['citations'][0]['passage_id'] in [p['passage_id'] for p in data['results']]
    with urlopen(Request(server+'/api/answer',data=json.dumps({'query':'water dosage','language':'en'}).encode())) as r:
        assert json.load(r)['qa']['status']=='UNANSWERABLE'


def test_real_endpoint_unavailable_and_host_guard(server):
    with pytest.raises(HTTPError) as e:
        urlopen(Request(server+'/api/retrieve', data=b'{"query":"wheat","language":"en"}'))
    assert e.value.code == 503
    with pytest.raises(HTTPError) as e:
        urlopen(Request(server, headers={'Host':'attacker.invalid'}))
    assert e.value.code == 403


def test_real_endpoint_uses_service_without_qa(server):
    # A synthetic service checks routing without models or copyrighted data.
    from unittest.mock import patch
    class Service:
        def search(self, payload):
            return {'mode':'local_e5', 'results':[], 'language':payload['language']}
    with patch.object(ThreadingHTTPServer, 'retrieval', Service(), create=True):
        for language in ('en','hi','hinglish'):
            with urlopen(Request(server+'/api/retrieve', data=json.dumps({'query':'synthetic', 'language':language}).encode())) as r:
                body=json.load(r)
                assert body['mode']=='local_e5' and body['language']==language and 'qa' not in body
    class Broken:
        def search(self, payload): raise RuntimeError('private source text')
    with patch.object(ThreadingHTTPServer, 'retrieval', Broken(), create=True):
        with pytest.raises(HTTPError) as e:
            urlopen(Request(server+'/api/retrieve', data=b'{"query":"synthetic","language":"en"}'))
        assert e.value.code==503 and b'private source' not in e.value.read()
