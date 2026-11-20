import pytest
from krishisaarthi.ui.availability import notices

@pytest.mark.parametrize('q',[
 "What is today's Ludhiana wheat price?", 'aaj gehu ka mandi bhav kya hai?',
 'आज गेहूँ का मंडी भाव क्या है?', 'current barley prices',
])
def test_live_prices(q):
    assert [x['code'] for x in notices(q)]==['live_prices']

@pytest.mark.parametrize('q',['Will the weather change tomorrow?', 'aaj mausam kaisa hai?', 'आज बारिश होगी?', 'Ludhiana weather forecast'])
def test_weather(q):
    assert [x['code'] for x in notices(q)]==['live_weather']

@pytest.mark.parametrize('q',[
 'Which weather conditions favour this disease?', 'How does the market price affect crop choice?',
 'When is barley watered?', 'current seed rate', 'Water the crop after how many days?',
 'मौसम की कौन सी परिस्थितियाँ अनुकूल हैं?',
])
def test_does_not_label_source_questions_unanswerable(q):
    assert notices(q)==[]


def test_both_capabilities_and_no_answerability_claim():
    result=notices('Current wheat prices and weather forecast')
    assert {x['code'] for x in result}=={'live_prices','live_weather'}
    assert all('answerability' not in x for x in result)


def test_filter_warning_without_switching_or_answerability_claim():
    r=notices('maize after wheat', 'wheat')
    assert 'crop_filter_check' in {n['code'] for n in r}
    assert notices('maize', 'maize')==[]
    assert notices('maize', 'all')==[]


def test_scope_and_treatment_notices():
    assert {n['code'] for n in notices('tomato in Rajasthan')}=={'outside_crop_scope','outside_region_scope'}
    assert 'treatment_evidence_check' in {n['code'] for n in notices('fungicide dose')}
    assert 'live_weather' in {n['code'] for n in notices('Will it rain tomorrow?')}
    assert notices('Rain during sowing')==[]
