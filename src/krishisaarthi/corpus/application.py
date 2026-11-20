"""AI-assisted application snapshots, separate from human-approved research corpora.

No gold labels or human approval are created here. Exact source intervals are
selected in a private plan; originals, canonical text and declared page files
remain mandatory. Existing research admission gates are not used or altered.
"""
import json
from collections import Counter
from pathlib import Path

from ..config import ChunkingConfig, fingerprint, text_hash
from ..evaluation.labels import validate_corpus, validate_span, validate_page_file
from ..ingestion.storage import read_jsonl, write_jsonl
from ..preprocessing.chunking import DeterministicChunker
from ..schemas import Document, EvidenceSpan, PageReference, Passage
from ..validation import require
from .workflow import private_path
import hashlib

POLICY = 'ai-assisted-local-application-v1'
CROPS = {'wheat': 'Wheat', 'barley': 'Barley', 'maize': 'Spring maize',
         'chickpea': 'Chickpea / gram', 'mustard': 'Rapeseed / mustard'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def construct(documents, inherited, plan):
    """Pure construction with source validation; returns no approval records."""
    require(plan['policy'] == POLICY, 'Wrong application collection policy')
    require(plan['human_verified'] is False, 'AI collection cannot claim human verification')
    docs = {d.document_id: d for d in documents}
    require(len(docs) == len(documents), 'Duplicate document IDs')
    if inherited:
        validate_corpus(documents, inherited)
    for d in documents:
        require(text_hash(d.canonical_text) == d.canonical_sha256, 'Canonical hash mismatch')
        require(sha(d.original_file_path) == d.content_sha256, 'Original hash mismatch')
        for p in d.pages:
            require(p.extracted_text_path is not None, 'Application pages require source files')
            validate_page_file(p, d)
    config = ChunkingConfig(size=256, overlap=48)
    chunker = DeterministicChunker(config)
    passages = list(inherited)
    regions = plan['regions']
    require(len({r['region_id'] for r in regions}) == len(regions), 'Duplicate source region')
    occupied = [(p.document_id, p.evidence_span.start_char, p.evidence_span.end_char) for p in inherited]
    for r in regions:
        require(r['crop'] in CROPS, 'Unsupported crop')
        require(r['curation'] == 'ai_assisted' and r['human_review'] == 'not_recorded', 'Invalid review attribution')
        span = EvidenceSpan.model_validate(r['evidence_span'])
        validate_span(span, docs)
        d = docs[span.document_id]
        pages = [p for p in d.pages if p.page_number == r['physical_page'] and p.canonical_start <= span.start_char < span.end_char <= p.canonical_end]
        require(len(pages) == 1, 'Region must belong to one exact source page')
        require(text_hash(d.canonical_text[span.start_char:span.end_char]) == r['text_sha256'], 'Region text hash mismatch')
        contexts = []
        for context in r.get('source_qualifications', []):
            ref = EvidenceSpan.model_validate(context['evidence_span'])
            validate_span(ref, docs)
            source = docs[ref.document_id]
            page = next((p for p in source.pages if p.page_number == context['physical_page']
                         and p.canonical_start <= ref.start_char < ref.end_char <= p.canonical_end), None)
            require(page is not None, 'Qualification page mismatch')
            text = source.canonical_text[ref.start_char:ref.end_char]
            require(text_hash(text) == context['text_sha256'], 'Qualification text hash mismatch')
            contexts.append({**context, 'text':text, 'filename':source.original_filename})
        for blocked in plan['excluded_spans']:
            require(not (blocked['document_id'] == span.document_id and blocked['start_char'] < span.end_char and blocked['end_char'] > span.start_char), 'Held/excluded span intersects selection')
        require(not any(doc == span.document_id and a < span.end_char and b > span.start_char for doc,a,b in occupied), 'Overlapping source selection')
        occupied.append((span.document_id, span.start_char, span.end_char))
        view = d.model_copy(update={'pages':[PageReference(page_number=r['physical_page'], canonical_start=span.start_char, canonical_end=span.end_char)]})
        for p in chunker.chunk(view):
            p.page_references = pages
            p.chunking_config_id = fingerprint({'base':chunker.config_id, 'policy':POLICY, 'region':r})
            p.passage_id = 'app-passage-' + fingerprint({'span':p.evidence_span.model_dump(), 'chunking':p.chunking_config_id})
            p.domain_metadata = {**d.domain_metadata, 'crop':r['crop'], 'region_id':r['region_id'],
                'topic':r['topic'], 'printed_page_label':r['printed_page_label'],
                'curation':'ai_assisted', 'human_review':'not_recorded', 'publication_allowed':False,
                'conditions_and_limitations':{'conditions':r['conditions'], 'remaining_concerns':r['limitations']},
                'source_region_start':span.start_char, 'source_region_end':span.end_char}
            if contexts:
                p.domain_metadata['source_qualifications'] = contexts
            passages.append(p)
    validate_corpus(documents, passages)
    require(len({text_hash(p.text) for p in passages}) == len(passages), 'Duplicate exact passage text')
    return passages


def build(documents, inherited, plan, output):
    output = private_path(Path(output))
    require(not output.exists(), 'Refusing to overwrite application snapshot')
    passages = construct(documents, inherited, plan)
    output.mkdir(mode=0o700, parents=True)
    write_jsonl(output/'documents.jsonl', documents)
    write_jsonl(output/'passages.jsonl', passages)
    (output/'selection.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    (output/'inherited.jsonl').write_text(''.join(p.model_dump_json()+'\n' for p in inherited))
    binding = {'policy':POLICY, 'selection':fingerprint(plan), 'documents':fingerprint([d.model_dump() for d in documents]),
               'passages':fingerprint([p.model_dump() for p in passages])}
    counts = dict(Counter(p.domain_metadata.get('crop','wheat') for p in passages))
    manifest = {'version':plan.get('version','Application Multicrop v1'), 'binding':binding, 'corpus_fingerprint':fingerprint(binding),
                'passage_count':len(passages), 'crop_counts':counts, 'document_count':len(documents),
                'chunking':{'size':256,'overlap':48}, 'human_verified':False,
                'notice':'AI-assisted application collection; inherited pilot review retained separately. Not a human-verified benchmark or current agricultural advice.'}
    if 'coverage_notes' in plan:
        manifest['coverage_notes'] = plan['coverage_notes']
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    names=['documents.jsonl','passages.jsonl','selection.json','inherited.jsonl','manifest.json']
    (output/'integrity.json').write_text(json.dumps({n:sha(output/n) for n in names},indent=2))
    for p in output.iterdir():p.chmod(0o600)
    return manifest


def load(directory):
    directory = private_path(Path(directory))
    ledger = json.loads((directory/'integrity.json').read_text())
    required={'documents.jsonl','passages.jsonl','selection.json','inherited.jsonl','manifest.json'}
    require(set(ledger) == required, 'Incomplete application integrity ledger')
    for name, expected in ledger.items():
        require(sha(directory/name) == expected, 'Application file changed')
    docs = read_jsonl(directory/'documents.jsonl',Document)
    passages = read_jsonl(directory/'passages.jsonl',Passage)
    inherited = read_jsonl(directory/'inherited.jsonl',Passage)
    plan = json.loads((directory/'selection.json').read_text())
    rebuilt = construct(docs,inherited,plan)
    require([p.model_dump() for p in passages] == [p.model_dump() for p in rebuilt], 'Application reconstruction mismatch')
    manifest = json.loads((directory/'manifest.json').read_text())
    binding={'policy':POLICY,'selection':fingerprint(plan),'documents':fingerprint([d.model_dump() for d in docs]),'passages':fingerprint([p.model_dump() for p in passages])}
    require(manifest['binding'] == binding and manifest['corpus_fingerprint'] == fingerprint(binding), 'Application fingerprint mismatch')
    require(manifest['passage_count'] == len(passages) and manifest['human_verified'] is False, 'Invalid application metadata')
    require(manifest['crop_counts'] == dict(Counter(p.domain_metadata.get('crop','wheat') for p in passages)), 'Crop counts mismatch')
    require(manifest.get('coverage_notes') == plan.get('coverage_notes'), 'Coverage metadata mismatch')
    return docs,passages,manifest
