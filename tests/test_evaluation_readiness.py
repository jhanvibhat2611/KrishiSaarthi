"""Synthetic fixtures only; no pilot labels or research evaluation."""
import pytest
from test_metrics import span, results
from krishisaarthi.schemas import GoldLabel, Query
from krishisaarthi.evaluation.metrics import complete_evidence_coverage_at_k, score_at_k
from krishisaarthi.evaluation.runner import evaluate
from krishisaarthi.config import EvaluationConfig, load_config
from krishisaarthi.retrieval.bm25 import BM25Retriever
from pathlib import Path


def test_all_spans_and_unchanged_first_evidence_rr():
    gold = GoldLabel(query_id='q', answerable=True, evidence_spans=[span(0,10), span(20,30)])
    hits = results(span(0,10), span(20,26), span(24,30))
    assert [complete_evidence_coverage_at_k(gold,hits,k) for k in [1,3,5]] == [0,1,1]
    assert score_at_k(gold,hits,1) == {'recall':.5,'reciprocal_rank':1}
    assert score_at_k(gold,hits,3) == {'recall':1,'reciprocal_rank':1}
    assert complete_evidence_coverage_at_k(gold,hits[:1],5) == 0


@pytest.mark.parametrize('hits,expected', [
    ([],0), ([span(0,6),span(4,10)],1), ([span(0,5),span(5,10)],1),
    ([span(0,4),span(5,10)],0), ([span(0,5),span(0,5)],0),
    ([span(0,10,'other')],0),
    ([span(0,10).model_copy(update={'canonical_sha256':'c'*64})],0),
    ([span(0,10).model_copy(update={'content_sha256':'c'*64})],0),
])
def test_strict_coverage_cases(hits,expected):
    gold=GoldLabel(query_id='q',answerable=True,evidence_spans=[span(0,10)])
    assert complete_evidence_coverage_at_k(gold,results(*hits),5)==expected


def test_metric_rejects_inapplicable_labels():
    for label in [GoldLabel(query_id='q',answerable=False),
                  GoldLabel(query_id='q',answerable=True,relevant_passage_ids=['p'],chunking_config_id='c')]:
        with pytest.raises(ValueError):complete_evidence_coverage_at_k(label,[],1)
    with pytest.raises(ValueError):
        complete_evidence_coverage_at_k(GoldLabel(query_id='q',answerable=True,evidence_spans=[span(0,1)]),[],0)


def test_languages_exclusions_and_no_positive_rates(document,passages):
    positive=next(p for p in passages if 'seedmarker' in p.text)
    qs=[Query(query_id='e',text='seedmarker',language='en'),
        Query(query_id='h',text='अज्ञातशब्द',language='hi'),
        Query(query_id='x',text='zzunknown',language='hinglish')]
    gold=[GoldLabel(query_id='e',answerable=True,evidence_spans=[positive.evidence_span]),
          GoldLabel(query_id='h',answerable=True,evidence_spans=[positive.evidence_span]),
          GoldLabel(query_id='x',answerable=False)]
    report=evaluate(BM25Retriever(),qs,gold,[document],passages,EvaluationConfig(ks=[1,3,5]))
    assert report['evaluated_queries']==2 and report['excluded_queries']==1
    assert report['per_language']['en']['evaluated_queries']==1
    assert report['per_language']['hi']['evaluated_queries']==1
    assert report['per_language']['hinglish']['metrics']['complete_evidence_coverage_at_k']=={'1':None,'3':None,'5':None}
    diag=report['diagnostics']
    assert diag['overall']['queries']==3
    assert diag['overall']['no_positive_match_count']==2
    assert diag['overall']['no_positive_match_rate']==pytest.approx(2/3)
    assert diag['per_language']['hi']['no_positive_match_rate']==1
    assert diag['per_language']['en']['no_positive_match_rate']==0
    assert next(d for d in diag['per_query'] if d['query_id']=='h')['zero_score_result_count']==min(5,len(passages))


def test_diagnostics_preserve_zero_ties_and_distinguish_overlap(passages):
    bm=BM25Retriever();bm.index(passages)
    q=Query(query_id='q',text='seedmarker')
    # Force matched term IDF zero/negative to exercise scores independently of overlap.
    for idf in [0,-1]:
        bm._model.idf['seedmarker']=idf
        before=bm.search(q,5)
        d=bm.diagnostics(q,5)
        assert d['vocabulary_overlap_count']==1 and not d['lexical_no_match']
        assert d['no_positive_match']
        assert bm.search(q,5)==before
        assert d['all_corpus_scores_zero']==(idf==0)
    empty=bm.diagnostics(Query(query_id='empty',text='   '),5)
    assert empty['empty_token_query'] and empty['lexical_no_match'] and empty['no_positive_match']


def test_pilot_configuration():
    c=load_config(Path('configs/pilot-evaluation.toml'))
    assert c.evaluation.ks==[1,3,5]
    assert c.chunking.size==256 and c.chunking.overlap==48
