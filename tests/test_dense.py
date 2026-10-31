"""Synthetic encoders only; no model downloads in unit tests."""
import json
from pathlib import Path
import numpy as np
import pytest
from krishisaarthi.schemas import Query
from krishisaarthi.retrieval.dense import DenseRetriever, TokenLimitError, normalized
from krishisaarthi.retrieval.e5 import E5Retriever
from krishisaarthi.retrieval.bge_m3 import BGEM3Retriever

SPECS=json.loads(Path('configs/dense-models.json').read_text())
class FakeEncoder:
    identity={'backend':'synthetic-test'}
    def __init__(self,dim):self.dim=dim;self.seen=[];self.count_override=None
    def token_counts(self,texts):return [self.count_override or len(t.split())+2 for t in texts]
    def encode(self,texts):
        self.seen.extend(texts)
        a=np.zeros((len(texts),self.dim));a[:,0]=3
        return a

@pytest.mark.parametrize('name,adapter',[('e5',E5Retriever),('bge-m3',BGEM3Retriever)])
def test_prefix_ranking_provenance_cache(name,adapter,passages,tmp_path):
    spec=SPECS[name];enc=FakeEncoder(spec['dimension']);r=adapter(enc,spec,'synthetic',tmp_path)
    before=[p.model_dump_json() for p in passages];r.index(list(reversed(passages)))
    assert enc.seen[0].startswith(spec['passage_prefix'])
    count=len(enc.seen);q=Query(query_id='q',text='गेहूँ wheat',language='hi');hits=r.search(q,3)
    assert enc.seen[-1]==spec['query_prefix']+q.text
    assert [h.passage_id for h in hits]==sorted(p.passage_id for p in passages)[:3]
    assert all(h.score==1 for h in hits)
    by={p.passage_id:p for p in passages}
    assert all(h.evidence_span==by[h.passage_id].evidence_span for h in hits)
    assert before==[p.model_dump_json() for p in passages]
    r2=adapter(FakeEncoder(spec['dimension']),spec,'synthetic',tmp_path);r2.index(passages)
    assert r2.cache_hit and not r2.encoder.seen
    # Changing a query's text with the same ID must not use its old embedding.
    r.search(Query(query_id='q',text='new text'),1);assert enc.seen[-1]==spec['query_prefix']+'new text'
    cache=tmp_path/r.cache_key/'embeddings.npy';cache.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='fingerprint'):r2.index(passages)

@pytest.mark.parametrize('name,adapter',[('e5',E5Retriever),('bge-m3',BGEM3Retriever)])
def test_token_boundary_counts_specials_and_prefix(name,adapter,passages):
    e=FakeEncoder(SPECS[name]['dimension']);r=adapter(e,SPECS[name],'synthetic')
    e.count_override=SPECS[name]['max_tokens'];r.index(passages)
    e.count_override+=1
    with pytest.raises(TokenLimitError):r.search(Query(query_id='long',text='long'),1)
    e.seen=[]
    with pytest.raises(TokenLimitError):r.index(passages)
    assert e.seen==[]

@pytest.mark.parametrize('array',[np.ones((1,3)),np.zeros((1,2)),np.array([[np.nan,1]]),np.array([[np.inf,1]])])
def test_bad_embeddings_rejected(array):
    with pytest.raises(ValueError):normalized(array,1,2)

def test_cosine_and_cache_binding(passages,tmp_path):
    e=FakeEncoder(2);spec=dict(SPECS['e5'],dimension=2)
    class DirectionEncoder(FakeEncoder):
        def encode(self,texts):return np.array([[3,4] if i==0 else [0,2] for i,_ in enumerate(texts)])
    r=DenseRetriever(DirectionEncoder(2),spec,'corpus-a',tmp_path);r.index(passages)
    hits=r.search(Query(query_id='q',text='x'),2)
    assert hits[0].score==pytest.approx(1) and hits[1].score==pytest.approx(.8)
    r2=DenseRetriever(FakeEncoder(2),spec,'corpus-b',tmp_path);r2.index(passages)
    assert r.cache_key!=r2.cache_key
    spec2=dict(spec,revision='c'*40);r3=DenseRetriever(FakeEncoder(2),spec2,'corpus-a',tmp_path);r3.index(passages)
    assert r.cache_key!=r3.cache_key

def test_wrong_recipe_rejected():
    with pytest.raises(ValueError):E5Retriever(FakeEncoder(768),dict(SPECS['e5'],query_prefix=''),'x')
    with pytest.raises(ValueError):BGEM3Retriever(FakeEncoder(1024),dict(SPECS['bge-m3'],pooling='mean'),'x')

def test_unindexed_and_invalid_inputs(passages):
    r=E5Retriever(FakeEncoder(768),SPECS['e5'],'x');q=Query(query_id='q',text='x')
    with pytest.raises(ValueError):r.search(q,1)
    with pytest.raises(ValueError):r.index([])
    with pytest.raises(ValueError):r.index([passages[0],passages[0]])
    with pytest.raises(ValueError):r.search(q,0)

@pytest.mark.parametrize('pooling,expected',[('mean',[.6,.8]),('cls',[1.,0.])])
def test_hf_pooling_masks_padding_without_truncation(pooling,expected):
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from krishisaarthi.retrieval.dense import HFEncoder
    class Tokenizer:
        def __call__(self,texts,**kw):
            assert kw['truncation'] is False and kw['add_special_tokens'] is True
            if kw.get('return_tensors')=='pt':
                return {'input_ids':torch.tensor([[1,2,0]]),'attention_mask':torch.tensor([[1,1,0]])}
            return {'input_ids':[[1,2]]}
    class Model:
        def __call__(self,**kw):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[3.,0.],[0.,4.],[999.,999.]]]))
    enc=HFEncoder.__new__(HFEncoder);enc.torch=torch;enc.spec={'pooling':pooling,'max_tokens':2};enc.device='cpu';enc.batch_size=1;enc.tokenizer=Tokenizer();enc.model=Model()
    assert enc.encode(['synthetic'])[0]==pytest.approx(expected)
    enc.spec['max_tokens']=1
    with pytest.raises(TokenLimitError):enc.encode(['synthetic'])


def test_mps_fallback_is_explicit_and_not_mixed():
    from krishisaarthi.retrieval.dense import HFEncoder,DeviceFallbackRequired
    enc=HFEncoder.__new__(HFEncoder);enc.device='mps'
    with pytest.raises(DeviceFallbackRequired):enc._device_error(RuntimeError('MPS out of memory'))
    with pytest.raises(RuntimeError,match='bad dimensions'):enc._device_error(RuntimeError('bad dimensions'))
    enc.device='cpu'
    with pytest.raises(RuntimeError,match='out of memory'):enc._device_error(RuntimeError('out of memory'))

def test_dense_evaluator_keeps_language_metrics_without_lexical_diagnostics(document,passages):
    from krishisaarthi.evaluation.runner import evaluate
    from krishisaarthi.config import EvaluationConfig
    from krishisaarthi.schemas import GoldLabel
    q=[Query(query_id='q',text='synthetic',language='hi'),Query(query_id='u',text='absent',language='hinglish')]
    first=sorted(passages,key=lambda p:p.passage_id)[0]
    labels=[GoldLabel(query_id='q',answerable=True,evidence_spans=[first.evidence_span]),GoldLabel(query_id='u',answerable=False)]
    r=E5Retriever(FakeEncoder(768),SPECS['e5'],'synthetic')
    report=evaluate(r,q,labels,[document],passages,EvaluationConfig(ks=[1,3,5]))
    assert report['per_language']['hi']['metrics']['Recall@1']==1
    assert report['per_language']['hinglish']['excluded_queries']==1
    assert not report['diagnostics']['available'] and report['diagnostics']['overall']['no_positive_match_rate'] is None
