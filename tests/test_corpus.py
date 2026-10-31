"""All source text and approvals in these tests are SYNTHETIC test fixtures."""
from pathlib import Path
import hashlib
import json
import pytest
from krishisaarthi.config import ChunkingConfig
from krishisaarthi.schemas import SourceMetadata, EvidenceSpan
from krishisaarthi.ingestion.storage import make_document, write_jsonl
from krishisaarthi.corpus.models import RegistryEntry, ReviewedRegion, ReviewChecks, Issue
from krishisaarthi.corpus.audit import extract, audit, propose_regions
from krishisaarthi.corpus.review import build_passages, review_fingerprint, validate_regions
from krishisaarthi.corpus.workflow import prepare, load_bundle, private_path
from conftest import FIXTURES


def entry(path, **overrides):
    values = dict(document_id='synthetic-doc', path=str(path.resolve()),
        content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), title='SYNTHETIC TEST',
        publication_type='fixture', publisher='Synthetic publisher', publication_date='2025-01',
        metadata_evidence_pages=[1], source_languages=['en'], printed_page_labels={'1': 'i'})
    values.update(overrides)
    return RegistryEntry(**values)


@pytest.fixture
def corpus_doc(tmp_path):
    path = tmp_path / 'synthetic.txt'
    text = 'Synthetic wheat marker one two three four five six seven eight nine ten.\n'
    path.write_text(text)
    return make_document(path, SourceMetadata(document_id='synthetic-doc'), [(1, text)], 'text/plain', 'test', '1')


def region(doc, start=0, end=None):
    return ReviewedRegion(region_id='synthetic-region',
        evidence_span=EvidenceSpan(document_id=doc.document_id, content_sha256=doc.content_sha256,
            canonical_sha256=doc.canonical_sha256, start_char=start, end_char=end or len(doc.canonical_text)),
        physical_page=1, printed_page_label='i', recommendation='candidate')


def approve(item):
    item = item.model_copy(deep=True)
    item.checks = ReviewChecks(**{name: True for name in ReviewChecks.model_fields if name != 'schema_version'})
    item.reviewer = 'synthetic-test-reviewer'
    item.reviewed_at = '2025-01-01T00:00:00+00:00'
    item.review_note = 'SYNTHETIC fixture approval only'
    item.reviewed_content_fingerprint = review_fingerprint(item)
    item.status = 'approved'
    return item


def test_pending_never_builds(corpus_doc):
    with pytest.raises(ValueError, match='No manually approved'):
        build_passages([corpus_doc], [region(corpus_doc)], [], ChunkingConfig())


def test_reviewed_regions_chunk_without_offset_changes(corpus_doc):
    r = region(corpus_doc, 10)
    r.section_boundaries = [corpus_doc.canonical_text.index('four')]
    r = approve(r)
    config = ChunkingConfig(size=5, overlap=1)
    before = corpus_doc.model_dump_json()
    chunks, selection = build_passages([corpus_doc], [r], [], config)
    again, repeat = build_passages([corpus_doc], [r], [], config)
    assert chunks == again and selection == repeat
    assert corpus_doc.model_dump_json() == before
    for p in chunks:
        a, b = p.evidence_span.start_char, p.evidence_span.end_char
        assert a >= 10
        assert p.text == corpus_doc.canonical_text[a:b]
        assert p.page_references == corpus_doc.pages
        assert p.domain_metadata['printed_page_label'] == 'i'
        assert p.domain_metadata['publication_allowed'] is False
        assert not a < r.section_boundaries[0] < b
        assert p.token_count <= 5


def test_disjoint_regions_never_join(corpus_doc):
    left, right = approve(region(corpus_doc, 0, 9)), approve(region(corpus_doc, 30))
    right.region_id = 'other'
    chunks, _ = build_passages([corpus_doc], [left, right], [], ChunkingConfig())
    assert all(p.evidence_span.end_char <= 9 or p.evidence_span.start_char >= 30 for p in chunks)


@pytest.mark.parametrize('change', ['missing_checks', 'stale_boundary', 'wrong_page', 'outside', 'overlap', 'missing_reviewer'])
def test_invalid_approvals(corpus_doc, change):
    r = approve(region(corpus_doc))
    rows = [r]
    if change == 'missing_checks': r.checks.privacy_clear = False
    elif change == 'stale_boundary': r.section_boundaries = [10]
    elif change == 'wrong_page': r.physical_page = 2
    elif change == 'outside': r.evidence_span.end_char += 10
    elif change == 'overlap':
        other = approve(region(corpus_doc, 10)); other.region_id = 'other'; rows.append(other)
    elif change == 'missing_reviewer': r.reviewer = None
    with pytest.raises(ValueError):
        build_passages([corpus_doc], rows, [], ChunkingConfig())


def test_cannot_remove_issue_ids_to_bypass_review(corpus_doc):
    r = approve(region(corpus_doc))
    issue = Issue(issue_id='synthetic-issue', code='privacy', document_id=corpus_doc.document_id,
                  physical_page=1, start_char=0, end_char=10, detail='Synthetic review flag')
    assert r.issue_ids == []
    with pytest.raises(ValueError, match='Unresolved'):
        build_passages([corpus_doc], [r], [issue], ChunkingConfig())
    r.resolutions = {'synthetic-issue': 'Synthetic source verified to contain no identifying information'}
    assert build_passages([corpus_doc], [r], [issue], ChunkingConfig())[0]


def test_original_and_canonical_mutation_rejected(corpus_doc):
    r = approve(region(corpus_doc))
    changed = corpus_doc.model_copy(update={'canonical_text': 'changed'})
    with pytest.raises(ValueError, match='Canonical'):
        validate_regions([changed], [r], [])
    Path(corpus_doc.original_file_path).write_text('changed')
    with pytest.raises(ValueError, match='Original'):
        validate_regions([corpus_doc], [r], [])


def test_non_english_cannot_be_approved(tmp_path):
    path = tmp_path / 'synthetic.txt'; path.write_text('गेहूँ wheat')
    doc = make_document(path, SourceMetadata(document_id='synthetic-doc'), [(1, path.read_text())], 'text/plain', 'test', '1')
    with pytest.raises(ValueError, match='Devanagari'):
        build_passages([doc], [approve(region(doc))], [], ChunkingConfig())


def test_audit_scopes_conflict_and_proposes_no_approvals(tmp_path):
    path = tmp_path / 'synthetic.txt'
    text = ('SYNTHETIC wheat paragraph with a harmless marker for software review only.\n\n'
            'On March 24, 2005, SYNTHETIC inconsistent date, not a fact.\n\n'
            'Haryana\nSYNTHETIC other wheat paragraph.\n')
    path.write_text(text)
    doc = make_document(path, SourceMetadata(document_id='synthetic-doc'), [(1, text)], 'text/plain', 'test', '1')
    # Page 1 is normally a cover; use physical page 2 for this synthetic body page.
    doc.pages[0].page_number = 2
    e = entry(path, known_issue_anchors=[{'physical_page':2, 'code':'date_conflict',
        'start_text':'On March 24, 2005,','end_before':'Haryana', 'detail':'Synthetic date conflict'}])
    pages, issues = audit([doc], [e], {doc.document_id:[{'physical_page':2,'extraction_error':None,'image_count':0}]})
    proposals = propose_regions([doc], [e], issues)
    assert all(r.status == 'pending' and not r.checks.privacy_clear for r in proposals)
    held = [r for r in proposals if any(i.code=='date_conflict' and i.issue_id in r.issue_ids for i in issues)]
    assert held
    assert any(r.recommendation=='candidate' and r.evidence_span.end_char <= text.index('On March') for r in proposals)
    assert doc.canonical_text == text


def test_privacy_and_table_diagnostics(tmp_path):
    path = tmp_path/'synthetic.txt'; text='Table 1: Name and Address of the Farmer\nSYNTHETIC PERSON S/o SYNTHETIC OTHER\n'
    path.write_text(text)
    doc = make_document(path, SourceMetadata(document_id='synthetic-doc'), [(2,text)], 'text/plain','test','1')
    _, issues = audit([doc],[entry(path)],{doc.document_id:[{'physical_page':2,'extraction_error':None,'image_count':0}]})
    assert {'privacy','table_alignment'} <= {i.code for i in issues}


def test_registry_does_not_infer_license():
    path = FIXTURES / 'synthetic_wheat.pdf'
    assert entry(path).license_status == 'unknown'
    with pytest.raises(ValueError, match='Verified license'):
        entry(path, license_status='verified')
    with pytest.raises(ValueError, match='Redistribution'):
        entry(path, redistribution_permission='verified')


def test_prepare_immutable_bundle_and_empty_templates(tmp_path):
    registry = tmp_path/'registry.jsonl'
    write_jsonl(registry, [entry(FIXTURES/'synthetic_wheat.pdf')])
    output = tmp_path/'bundle'
    report = prepare(registry, output)
    assert report['approved_passages'] == 0 and not report['failures']
    docs, regions, issues, integrity = load_bundle(output)
    assert all(r.status == 'pending' for r in regions)
    for name in ['questions.csv','evidence.csv']:
        assert len((output/'annotations'/name).read_text().splitlines()) == 1
    assert not (output/'approved-passages.jsonl').exists()
    with pytest.raises(FileExistsError): prepare(registry, output)
    (output/'issues.jsonl').write_text('')
    with pytest.raises(ValueError, match='Immutable'):
        load_bundle(output)


def test_extraction_failure_is_reported(tmp_path):
    bad=tmp_path/'bad.pdf'; bad.write_bytes(b'not a PDF')
    registry=tmp_path/'registry.jsonl'; write_jsonl(registry,[entry(bad)])
    report=prepare(registry,tmp_path/'bundle')
    assert report['failures'][0]['status']=='hold'
    assert report['documents'][0]['pages']==0


def test_page_extraction_failure_and_scan_candidate(tmp_path, monkeypatch):
    class FailedPage:
        images=[]
        def extract_text(self): raise ValueError('synthetic failure')
    class ImagePage:
        images=[object()]
        def extract_text(self): return ''
    class Reader:
        is_encrypted=False
        pages=[FailedPage(), ImagePage()]
    monkeypatch.setattr('krishisaarthi.corpus.audit.pypdf.PdfReader',lambda path: Reader())
    path=tmp_path/'synthetic.pdf'; path.write_bytes(b'SYNTHETIC placeholder')
    e=entry(path); doc, diag=extract(e,path)
    assert diag[0]['extraction_error']=='ValueError'
    pages,issues=audit([doc],[e],{doc.document_id:diag})
    assert len(pages)==2
    assert {'extraction_error','no_text_scan_candidate'} <= {i.code for i in issues}
    assert propose_regions([doc],[e],issues)==[]


def test_private_output_guard(tmp_path):
    with pytest.raises(ValueError,match='ignored repository data'):
        private_path(tmp_path/'public')


def test_duplicate_content_is_flagged(tmp_path):
    text='SYNTHETIC wheat marker repeated for a software diagnostic. '*8
    path=tmp_path/'synthetic.txt'; path.write_text(text)
    doc=make_document(path,SourceMetadata(document_id='synthetic-doc'),[(2,text),(3,text)],'text/plain','test','1')
    _,issues=audit([doc],[entry(path)],{doc.document_id:[{'physical_page':i,'extraction_error':None,'image_count':0} for i in [2,3]]})
    assert sum(i.code=='duplicate_candidate' for i in issues)==2


def test_short_fragments_are_held_and_proposals_repeat(tmp_path):
    path=tmp_path/'synthetic.txt'; path.write_text('Haryana\n\nTiny fragment\n')
    doc=make_document(path,SourceMetadata(document_id='synthetic-doc'),[(2,path.read_text())],'text/plain','test','1')
    e=entry(path)
    _,issues=audit([doc],[e],{doc.document_id:[{'physical_page':2,'extraction_error':None,'image_count':0}]})
    first=propose_regions([doc],[e],issues)
    count=len(issues)
    second=propose_regions([doc],[e],issues)
    assert first==second and len(issues)==count
    assert all(r.status=='pending' and r.recommendation=='hold' for r in first)


def test_cli_inspect_and_fail_closed_build(tmp_path, monkeypatch, capsys):
    import sys
    from krishisaarthi.corpus.__main__ import main
    registry=tmp_path/'registry.jsonl'; write_jsonl(registry,[entry(FIXTURES/'synthetic_wheat.pdf')])
    bundle=tmp_path/'bundle'; prepare(registry,bundle)
    docs,regions,issues,_=load_bundle(bundle)
    # Storage protection is tested independently; exercise CLI in pytest's temp root.
    monkeypatch.setattr('krishisaarthi.corpus.__main__.private_path',lambda path:path.resolve())
    monkeypatch.setattr(sys,'argv',['corpus','inspect','--bundle',str(bundle),'--region-id',regions[0].region_id])
    main()
    report=json.loads(capsys.readouterr().out)
    assert 'unreviewed_local_source_text' not in report
    assert report['region']['status']=='pending'
    output=tmp_path/'index'
    monkeypatch.setattr(sys,'argv',['corpus','build','--bundle',str(bundle),'--output',str(output)])
    with pytest.raises(SystemExit) as exc: main()
    assert exc.value.code==2 and not output.exists()
    # Only synthetic test data are approved here; never real corpus approvals.
    approved=approve(regions[0])
    approved.resolutions={i.issue_id:'SYNTHETIC test-only resolution' for i in issues
        if i.document_id==approved.evidence_span.document_id and
        i.start_char<approved.evidence_span.end_char and i.end_char>approved.evidence_span.start_char}
    write_jsonl(bundle/'regions.jsonl',[approved])
    main()
    assert (output/'approved-passages.jsonl').exists()
    assert json.loads((output/'index-manifest.json').read_text())['labels_generated'] is False


@pytest.mark.parametrize('date,precision,valid', [('2023','year',True),('2023-08','month',True),('2023','month',False),('2023-08','year',False),('2023-13','month',False)])
def test_registry_date_precision(date, precision, valid):
    if valid:
        assert entry(FIXTURES/'synthetic_wheat.pdf', publication_date=date, date_precision=precision)
    else:
        with pytest.raises(ValueError):
            entry(FIXTURES/'synthetic_wheat.pdf', publication_date=date, date_precision=precision)


def test_selected_audit_preserves_full_canonical_source(tmp_path):
    path = FIXTURES/'synthetic_wheat.pdf'
    e = entry(path, audit_page_numbers=[2], geographic_applicability=['Synthetic region'])
    original, _ = extract(e, path)
    registry = tmp_path/'registry.jsonl'; write_jsonl(registry, [e])
    report = prepare(registry, tmp_path/'bundle')
    docs, regions, issues, _ = load_bundle(tmp_path/'bundle')
    assert docs[0].canonical_text == original.canonical_text
    assert docs[0].canonical_sha256 == original.canonical_sha256
    assert len(docs[0].pages) == len(original.pages)
    assert {p['physical_page'] for p in report['pages']} == {2}
    assert all(r.physical_page == 2 for r in regions)
    assert all(i.physical_page == 2 for i in issues)
    assert docs[0].domain_metadata['geographic_applicability'] == ['Synthetic region']
    validate_regions(docs, regions, issues)


@pytest.mark.parametrize('pages', [[], [0], [2,1], [1,1], [999]])
def test_invalid_audit_selection(tmp_path, pages):
    with pytest.raises(ValueError):
        e = entry(FIXTURES/'synthetic_wheat.pdf', audit_page_numbers=pages)
        registry=tmp_path/'registry.jsonl'; write_jsonl(registry,[e])
        prepare(registry,tmp_path/'bundle')


def test_null_glyph_is_held_without_text_repair(tmp_path):
    path=tmp_path/'synthetic.txt'; text='SYNTHETIC wheat noti\x00cation text with enough words for review.'
    path.write_text(text)
    doc=make_document(path,SourceMetadata(document_id='synthetic-doc'),[(2,text)],'text/plain','test','1')
    e=entry(path)
    _,issues=audit([doc],[e],{doc.document_id:[{'physical_page':2,'extraction_error':None,'image_count':0}]})
    assert any(i.code=='control_characters' for i in issues)
    assert all(r.recommendation=='hold' for r in propose_regions([doc],[e],issues))
    assert doc.canonical_text == text
