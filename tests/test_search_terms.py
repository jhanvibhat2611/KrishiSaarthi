"""Synthetic query vocabulary tests, without source data or model loading."""
import pytest
from krishisaarthi.ui.hinglish import normalize
from krishisaarthi.ui.search_terms import expand
from krishisaarthi.ui.server import validate_payload

@pytest.mark.parametrize('q,term',[
 ('When is a fictional crop watered?','irrigation'),
 ('Does this sample need watering?','irrigation'),
 ('When was this fictional crop planted?','sowing'),
 ('When is the sample harvested?','harvesting'),
 ('Which fertiliser is mentioned?','fertilizer'),
 ('What seed quantity is recorded?','seed rate'),
])
def test_expansion_retains_question(q,term):
    r=expand(normalize(q,'en'),'en')
    assert r['search_query']==q+'\nRelated search terms: '+term
    assert r['original_query']==q and r['expanded_terms']==[term]

@pytest.mark.parametrize('q',[
 'water quality', 'cut a sample', 'plant disease', 'waterlogged soil',
 'irrigation and watering', 'PBW 869 20 kg 10% nitrogen',
])
def test_ambiguous_terms_and_already_present_terms_are_unchanged(q):
    assert expand(normalize(q,'en'),'en')['search_query']==q


def test_negation_quantities_and_logic_are_not_rewritten():
    q='Should the sample not be watered before 15 days or after 20 days?'
    r=expand(normalize(q,'en'),'en')
    assert r['search_query'].startswith(q+'\n')
    assert r['expanded_terms']==['irrigation']


def test_disable_and_hindi_leave_query_unchanged():
    q='watered planted'
    assert expand(normalize(q,'en'),'en',False)['search_query']==q
    assert expand(normalize(q,'hi'),'hi')['search_query']==q


def test_boundaries_case_and_multiple_terms():
    q='WATERED and PLANTED'
    assert expand(normalize(q,'en'),'en')['expanded_terms']==['irrigation','sowing']
    assert not expand(normalize('unwatered plantedness','en'),'en')['applied']


def test_reject_non_boolean():
    for value in [None,0,'false',[]]:
        with pytest.raises(ValueError):validate_payload({'query':'test','language':'en','expand_terms':value})

@pytest.mark.parametrize('q,term',[
 ('How many waterings for a sample?', 'irrigation'),
 ('What row gap is recorded?', 'row spacing'),
 ('Distance between the plants?', 'plant spacing'),
])
def test_additional_inflections_and_spacing(q,term):
    r=expand(normalize(q,'en'),'en')
    assert term in r['expanded_terms']
    assert r['search_query'].startswith(q+'\n')
