"""No model initialization, real searches or real generations in these regressions."""
import json
from pathlib import Path
import pytest
from krishisaarthi.evaluation.qa_journal import AttemptJournal
from krishisaarthi.evaluation.qa_runtime import validate_complete_documents
from krishisaarthi.evaluation.qa_adapters import PairedComparison
from krishisaarthi.evaluation.qa_comparison_fake import Question
from test_qa_adapters import MockDecoder,proposal,evidence


def test_full_document_metadata_mutations_rejected(document):
    validate_complete_documents([document],[document])
    for update in [{'domain_metadata':{'crop':'different','license_status':'invented'}},
                   {'source_url':'https://wrong.invalid'},{'extraction_metadata':{'version':'changed'}}]:
        with pytest.raises(ValueError,match='metadata'):
            validate_complete_documents([document.model_copy(update=update)],[document])
    with pytest.raises(ValueError):validate_complete_documents([document],None)


def test_interrupted_attempt_permanently_consumed(tmp_path):
    j=AttemptJournal(tmp_path/'journal',[('case','direct')]);j.before('case','direct')
    reopened=AttemptJournal(tmp_path/'journal',[('case','direct')])
    with pytest.raises(FileExistsError):reopened.before('case','direct')
    j.after({'case_id':'case','arm':'direct','status':'error','raw_output':'synthetic partial'})
    with pytest.raises(FileExistsError):reopened.before('case','direct')
    data=json.loads((tmp_path/'journal/case--direct.json').read_text())
    assert data['attempt_consumed'] and data['record']['raw_output']=='synthetic partial'


def test_journal_before_decode_and_after(document,passages,tmp_path):
    path=tmp_path/'journal';slots=[('pilot-v01-in-06-en',a) for a in ('direct','rag')]
    class Observing(MockDecoder):
        def decode(self,*args):
            arm='direct' if not self.calls else 'rag'
            assert json.loads((path/f'pilot-v01-in-06-en--{arm}.json').read_text())['state']=='reserved_consumed'
            return super().decode(*args)
    d=Observing();r=PairedComparison(d,proposal(),AttemptJournal(path,slots));e=evidence(document,passages)
    out=r.pair(slots[0][0],Question(text=e.question.text,language='en'),e,[document])
    assert out['ordering']=='fixed_direct_first' and len(d.calls)==2
    assert all(json.loads(f.read_text())['state']=='completed_record' for f in path.glob('*.json'))


def test_reason_codes_and_no_consumption_on_overflow(document,passages,tmp_path):
    d=MockDecoder();d.prompt_tokens=lambda _: [1]*7809
    j=AttemptJournal(tmp_path/'journal',[('pilot-v01-in-06-en',a) for a in ('direct','rag')])
    r=PairedComparison(d,proposal(),j);e=evidence(document,passages)
    out=r.pair('pilot-v01-in-06-en',Question(text=e.question.text,language='en'),e,[document])
    assert out['rows'][0]['failure_category']=='context_overflow'
    assert not list((tmp_path/'journal').iterdir()) and not d.calls


def test_mutated_pack_qualifications_block_before_decode(document,passages):
    d=MockDecoder();r=PairedComparison(d,proposal());e=evidence(document,passages)
    e.document_metadata={document.document_id:{'geography':'invented'}}
    out=r.pair('pilot-v01-in-06-en',Question(text=e.question.text,language='en'),e,[document])
    assert out['rows'][0]['failure_category']=='source_validation_failure' and not d.calls


def test_execution_rejects_wrong_approval_without_loading(tmp_path):
    from krishisaarthi.evaluation.qa_real import execute
    f=tmp_path/'approval.txt';f.write_text('not authorized')
    with pytest.raises(ValueError,match='Approval'):execute(tmp_path,tmp_path/'data/run',f)
    assert not (tmp_path/'data').exists()
