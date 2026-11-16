"""Engineering tests only; no real-data/model performance claims."""
import pytest
from pydantic import ValidationError
from krishisaarthi.ui.qa import QUESTIONS, ANSWERS, answer_question, Draft, Response
from krishisaarthi.ui.server import search

def run(topic='soil',language='en',generator=None):
    q=QUESTIONS[topic][language]
    return answer_question(q,language,search({'query':q,'language':language})['results'],generator)

@pytest.mark.parametrize('topic',['soil','seed'])
@pytest.mark.parametrize('language',['en','hi','hinglish'])
def test_supported_multilingual(topic,language):
    r=run(topic,language)
    assert r.status=='review_required' and r.answer==ANSWERS[topic][language]
    assert r.citations and r.language==language
    assert r.mode=='synthetic_test_double' and r.confidence=='not_estimated'
    assert r.factual_verification=='not_independently_verified'

@pytest.mark.parametrize('query',['What fertilizer dose should I use?','सिंचाई कितनी बार करें?','Pesticide kitna dena hai?'])
def test_out_of_scope(query):
    r=answer_question(query,'en',[])
    assert r.status=='UNANSWERABLE' and r.answer is None and not r.citations
    assert r.answerability=='insufficient_demo_evidence'

def test_missing_and_tampered_evidence():
    q=QUESTIONS['soil']['en']
    assert answer_question(q,'en',[]).status=='UNANSWERABLE'
    evidence=search({'query':q,'language':'en'})['results']
    evidence[0]['text']='SYNTHETIC altered source'
    assert answer_question(q,'en',evidence).status=='UNANSWERABLE'

class Adapter:
    name='test-adapter'
    def __init__(self,output):self.output=output
    def generate(self,**kwargs):
        assert set(kwargs)=={'language','fixture','evidence'}
        if isinstance(self.output,Exception):raise self.output
        return self.output

@pytest.mark.parametrize('output,category',[
    ({'status':'draft','answer':'x','evidence_handles':[]},'malformed_output'),
    ({'status':'draft','answer':'x'},'malformed_output'),
    ('not json','malformed_output'),
    ({'status':'draft','answer':'x','evidence_handles':['E99']},'invalid_citation'),
    ({'status':'draft','answer':'x','evidence_handles':['E1;bad']},'invalid_citation'),
    ({'status':'draft','answer':'unsupported claim','evidence_handles':['E1']},'unsupported_fixture_output'),
    (RuntimeError('private failure detail'),'generator_failure'),
])
def test_failures_are_not_abstentions(output,category):
    r=run(generator=Adapter(output))
    assert r.status=='error' and r.failure_category==category
    assert r.answerability=='not_established' and r.answer is None and not r.citations
    assert 'private failure detail' not in r.model_dump_json()

def test_adapter_abstention():
    r=run(generator=Adapter({'status':'insufficient_evidence','answer':'','evidence_handles':[]}))
    assert r.status=='UNANSWERABLE' and r.answer is None

def test_contract_rejects_uncited_answer_and_bad_language():
    with pytest.raises(ValidationError):Draft(status='draft',answer='x',evidence_handles=[])
    with pytest.raises(ValidationError):answer_question('x','fr',[])
    with pytest.raises(ValidationError):Response(original_question='x',language='en',query_type='outside_fixture_scope',status='error',answerability='insufficient_demo_evidence',reason='x')
