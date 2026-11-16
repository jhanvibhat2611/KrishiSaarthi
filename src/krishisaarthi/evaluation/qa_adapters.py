"""QA-C2 adapters. Importing this module never loads models or reads research data.

No executable real-run entry point. Callers must obtain separate run authorization.
"""
import json
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Protocol
from .qa_comparison_fake import Question, build_direct_prompt, RAG_SYSTEM, parse_output, empty_human_review, fingerprint
from .qa_readiness import E5ComparisonProposal
from ..schemas import Query
from ..evaluation.labels import validate_corpus
from ..rag.models import EvidencePack
from ..rag.handles import HandleAnswer, HandleClaim, resolve_answer, handle_table
from ..rag.evidence import CitationValidator


def require(ok, message):
    if not ok: raise ValueError(message)


class Decoder(Protocol):
    execution_kind: str
    last_output_text: str
    last_metrics: dict
    def prompt_tokens(self, messages): ...
    def decode(self, messages, max_tokens): ...


class QwenDecoder:
    """Wrap an already loaded MLXTextBackend; never load/download a checkpoint here."""
    execution_kind = 'real_model'

    def __init__(self, backend, proposal):
        # Import is safe: MLX itself is imported only by the existing constructor/decode.
        from ..rag.two_stage import MLXTextBackend
        require(isinstance(backend, MLXTextBackend), 'Expected existing MLXTextBackend')
        self.proposal = E5ComparisonProposal.model_validate(proposal.model_dump(), strict=True)
        config = backend.config
        require(config['model_id'] == proposal.model_id and config['revision'] == proposal.model_revision
                and config['tokenizer_revision'] == proposal.tokenizer_revision, 'Pinned checkpoint mismatch')
        require(config['packages'] == proposal.generation_packages, 'Package configuration mismatch')
        inference = config['inference']
        for field, expected in [('temperature',0.0),('seed',0),('batch_size',1),('context_limit',8192),
                                ('prompt_prefill_step_size',256),('chat_template',proposal.chat_template)]:
            require(inference[field] == expected, 'Decoding configuration mismatch')
        self.backend = backend
        self.config_fingerprint = fingerprint(config)

    @property
    def last_output_text(self): return self.backend.last_output_text
    @property
    def last_metrics(self): return self.backend.last_metrics

    def prompt_tokens(self, messages):
        return self.backend.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def decode(self, messages, max_tokens):
        self.backend.last_metrics = {}
        self.backend.last_output_text = ''
        require(max_tokens == 384, 'Only the declared output budget is supported')
        require(fingerprint(self.backend.config)==self.config_fingerprint,'Decoder configuration changed')
        return self.backend.decode(messages, max_tokens)


class E5EvidenceAdapter:
    """Use the existing loaded read-only UI service; never call index/build here."""
    def __init__(self, local_service, proposal):
        self.proposal = E5ComparisonProposal.model_validate(proposal.model_dump(), strict=True)
        r = local_service.retriever
        require(r.cache_hit and r.cache_key == proposal.embedding_cache_key
                and r.model_fingerprint == proposal.retrieval_model_fingerprint
                and r.corpus_fingerprint == proposal.corpus_fingerprint, 'Frozen E5 binding mismatch')
        self.service = local_service

    def retrieve(self, question):
        question = Question.model_validate(question.model_dump(), strict=True)
        result = self.service.search({'query':question.text, 'language':question.language})
        require(result['mode'] == 'local_e5' and result['corpus_fingerprint'] == self.proposal.corpus_fingerprint,
                'Wrong retrieval collection')
        hits = result['results']
        require(len(hits) == 5 and len({h['passage_id'] for h in hits}) == 5, 'Expected five unique E5 results')
        selected = []
        for rank, hit in enumerate(hits, 1):
            p = self.service.passages.get(hit['passage_id'])
            require(p is not None and hit['rank'] == rank and hit['document_id'] == p.document_id
                    and hit['text'] == p.text and hit['evidence_span'] == p.evidence_span.model_dump(),
                    'Retrieved source mismatch')
            selected.append(p.model_copy(deep=True))
        documents = list(self.service.documents.values())
        validate_corpus(documents, selected)
        return EvidencePack(question=Query(query_id='comparison-request',text=question.text,language=question.language),
            passages=selected, retriever_id=self.service.retriever.retriever_id,
            document_metadata={d.document_id:d.domain_metadata for d in documents})


def rag_prompt(question, evidence):
    require(evidence.question.text == question.text and evidence.question.language == question.language,
            'Question/evidence binding mismatch')
    require(len({p.passage_id for p in evidence.passages}) == len(evidence.passages), 'Duplicate evidence')
    keys = ('publication_date','geographic_applicability','applicability','conditions_and_limitations','context_references')
    sources = [{'handle':h, 'text':p.text,
                'scope':{k:v for k,v in {**evidence.document_metadata.get(p.document_id,{}),**p.domain_metadata}.items() if k in keys}}
               for h,p in handle_table(evidence).items()]
    return [{'role':'system','content':RAG_SYSTEM},
            {'role':'user','content':json.dumps({'question':question.model_dump(),'sources':sources},ensure_ascii=False)}]


def validate_output(raw, arm, evidence):
    output = parse_output(raw, arm)  # Existing duplicate-key/schema/status checks.
    citations = []
    if arm == 'rag' and output.status == 'draft':
        answer = resolve_answer(HandleAnswer(answer_status='draft',answer_text='\n'.join(c.text for c in output.claims),
            claims=[HandleClaim(**c.model_dump()) for c in output.claims],uncertainty=[]),evidence)
        require(not CitationValidator().validate(answer.claims,evidence), 'Resolved citation mismatch')
        byid = {p.passage_id:p for p in evidence.passages}
        for i, claim in enumerate(answer.claims):
            for citation in claim.citations:
                p = byid[citation.passage_id]
                citations.append({'claim_index':i, **citation.model_dump(), 'document_id':p.document_id,
                                  'evidence_span':p.evidence_span.model_dump(),
                                  'pages':[x.model_dump() for x in p.page_references],
                                  'source_url':p.source_url,'original_filename':p.original_filename,
                                  'qualifications':p.domain_metadata})
    return {'status':'review_required' if output.status == 'draft' else output.status,
            'claims':[c.model_dump() for c in output.claims], 'reason':output.reason,
            'citations':citations, 'factual_support':'not_verified'}


class PairedComparison:
    """One decode per arm, fixed planned denominators; injectable decoder only.

    Future orchestration supplies provenance-validated evidence before calling pair.
    This class has no dataset/reference loader, retriever invocation or model loader.
    """
    def __init__(self, decoder, proposal, journal=None):
        self.proposal = E5ComparisonProposal.model_validate(proposal.model_dump(),strict=True)
        require(decoder.execution_kind in ('mock','real_model'), 'Declare execution kind')
        if decoder.execution_kind == 'real_model':
            require(isinstance(decoder,QwenDecoder), 'Real calls require pinned Qwen adapter')
        self.journal = journal
        self.decoder = decoder
        self.rows = {(c.case_id,arm):{'case_id':c.case_id,'group_id':c.group_id,'language':c.language,'arm':arm,
            'execution_kind':decoder.execution_kind,'attempted':False,'status':'not_attempted','failure_category':None,
            'raw_output':'','input_tokens':None,'output_tokens':None,'latency_seconds':None,
            'response_completed':False,'human_review':empty_human_review(),
            'citation_human_review':None,'citation_applicability':'not_applicable' if arm=='direct' else 'applicable',
            'externally_verified_false':None,'external_verification_reference':None,'reference_scoring_enabled':False}
            for c in proposal.cases for arm in ('direct','rag')}
        self.used = set()
        self.halted = False

    def pair(self, case_id, question, evidence, documents, *, frozen_passages=None):
        require(not self.halted and case_id not in self.used and (case_id,'direct') in self.rows, 'Unavailable or repeated planned case')
        question = Question.model_validate(question.model_dump(),strict=True)
        require(question.language == self.rows[(case_id,'direct')]['language'], 'Case language mismatch')
        self.used.add(case_id)
        preflight_code='source_validation_failure'
        try:
            require(evidence.document_metadata=={d.document_id:d.domain_metadata for d in documents},'Document metadata mismatch')
            validate_corpus(documents,evidence.passages)  # No synthetic zero-based citation substitution.
            require(0 < len(evidence.passages) <= 5,'Invalid evidence count')
            if self.decoder.execution_kind == 'real_model':
                require(frozen_passages is not None and len(frozen_passages)==15 and
                        fingerprint([p.model_dump() for p in frozen_passages])=='757436adcc6bb0244f92daaac56888637bd679ae2bcbd4e8dc5c8d518ebb6125',
                        'Require exact frozen pilot records')
                frozen={p.passage_id:p.model_dump() for p in frozen_passages}
                require(all(frozen.get(p.passage_id)==p.model_dump() for p in evidence.passages),'Evidence outside frozen snapshot')
                require(len(evidence.passages)==5 and evidence.retriever_id==self.proposal.index_identifier,'Require frozen E5 Top-5')
            prompts = {'direct':build_direct_prompt(question),'rag':rag_prompt(question,evidence)}
            preflight_code='tokenization_failure'
            tokens = {arm:self.decoder.prompt_tokens(messages) for arm,messages in prompts.items()}
            for arm, ids in tokens.items():
                require(isinstance(ids,list) and bool(ids) and all(type(x) is int for x in ids),'Invalid token accounting')
                row = self.rows[(case_id,arm)]
                row.update(prompt=prompts[arm],prompt_sha256=fingerprint(prompts[arm]),input_tokens=len(ids),
                           prompt_tokens_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),output_token_cap=384,
                           evidence_fingerprint=evidence.fingerprint if arm=='rag' else None)
            preflight_code='context_overflow'
            require(all(len(ids)+384<=8192 for ids in tokens.values()),'Prompt budget exceeded')
        except Exception:
            for arm in ('direct','rag'):
                self.rows[(case_id,arm)].update(status='blocked',failure_category=preflight_code)
            return self.result()
        for arm in ('direct','rag'):
            row = self.rows[(case_id,arm)]
            if self.journal is not None:self.journal.before(case_id,arm)
            row['attempted'] = True
            started = time.perf_counter()
            try:
                raw = self.decoder.decode(prompts[arm],384)
                row['raw_output'] = raw
                row['response_completed'] = True
                row.update(validate_output(raw,arm,evidence if arm=='rag' else None))
            except Exception as exc:
                category = getattr(exc,'category','model_runtime_error')
                allowed = {'invalid_json','schema_failure','invalid_status','incorrect_status','invalid_citation',
                           'output_length_exhaustion','repetition','model_runtime_error'}
                row.update(status='error',failure_category=category if category in allowed else 'model_runtime_error')
                if row['failure_category']=='model_runtime_error': self.halted=True
            finally:
                row['latency_seconds'] = time.perf_counter()-started
                if not row['response_completed']: row['raw_output'] = self.decoder.last_output_text
                metrics = dict(self.decoder.last_metrics)
                row['backend_metrics'] = metrics
                if self.decoder.execution_kind=='real_model':
                    required=('input_tokens','output_tokens','runtime_seconds','mlx_peak_bytes','process_lifetime_peak_rss_bytes','finish_reason','prompt_tokens_sha256')
                    if any(metrics.get(k) is None for k in required):
                        row.update(status='error',failure_category='missing_telemetry');self.halted=True
                    for k in ('runtime_seconds','mlx_peak_bytes','process_lifetime_peak_rss_bytes'):
                        v=metrics.get(k)
                        if not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
                            row.update(status='error',failure_category='missing_telemetry');self.halted=True
                    if metrics.get('mlx_peak_bytes',0)>6*1024**3 or metrics.get('process_lifetime_peak_rss_bytes',0)>8*1024**3:
                        row.update(status='error',failure_category='resource_limit');self.halted=True
                count = metrics.get('output_tokens')
                if count is not None and (type(count) is not int or not 0<=count<=384):
                    row.update(status='error',failure_category='telemetry_failure');self.halted=True
                else: row['output_tokens']=count
                if row['response_completed'] and (count is None or
                        not isinstance(metrics.get('runtime_seconds'),(int,float)) or
                        not math.isfinite(metrics['runtime_seconds']) or metrics['runtime_seconds']<0):
                    row.update(status='error',failure_category='telemetry_failure');self.halted=True
                if metrics.get('prompt_tokens_sha256',row['prompt_tokens_sha256'])!=row['prompt_tokens_sha256']:
                    row.update(status='error',failure_category='telemetry_failure');self.halted=True
                if 'input_tokens' in metrics and metrics['input_tokens']!=row['input_tokens']:
                    row.update(status='error',failure_category='telemetry_failure');self.halted=True
            if row['status']=='error':
                row['claims']=[];row['citations']=[]
            if self.journal is not None:self.journal.after(row)
            if self.halted: break
        return self.result()

    def result(self):
        rows=list(self.rows.values())
        def counts(items):
            return {'planned':len(items),'attempted':sum(r['attempted'] for r in items),
                    'completed_responses':sum(r['response_completed'] for r in items),
                    'answer_drafts':sum(r['status']=='review_required' for r in items),
                    'withholding':sum(r['status'] in ('insufficient','uncertain') for r in items),
                    'failed':sum(r['status']=='error' for r in items),
                    'unattempted':sum(not r['attempted'] for r in items),
                    'observed_output_tokens':sum(r['output_tokens'] for r in items if r['output_tokens'] is not None),
                    'output_token_observation_count':sum(r['output_tokens'] is not None for r in items),
                    'failure_categories':{category:sum(r['failure_category']==category for r in items)
                                          for category in sorted({r['failure_category'] for r in items if r['failure_category']})}}
        return {'execution_kind':self.decoder.execution_kind,'proposal_sha256':fingerprint(self.proposal.model_dump()),
                'configuration':self.proposal.model_dump(),'ordering':'fixed_direct_first',
                'mock_decode_attempts':sum(r['attempted'] for r in rows) if self.decoder.execution_kind=='mock' else 0,
                'model_attempts':sum(r['attempted'] for r in rows) if self.decoder.execution_kind=='real_model' else 0,
                'reference_scoring_enabled':False,'overall':counts(rows),
                'by_arm':{a:counts([r for r in rows if r['arm']==a]) for a in ('direct','rag')},
                'by_language':{lang:{a:counts([r for r in rows if r['language']==lang and r['arm']==a])
                                          for a in ('direct','rag')} for lang in ('en','hi','hinglish')},
                'paired_prompt_tokens':[{'case_id':c.case_id,
                    'direct':self.rows[(c.case_id,'direct')]['input_tokens'],
                    'rag':self.rows[(c.case_id,'rag')]['input_tokens'],
                    'rag_minus_direct':None if any(self.rows[(c.case_id,a)]['input_tokens'] is None for a in ('direct','rag'))
                      else self.rows[(c.case_id,'rag')]['input_tokens']-self.rows[(c.case_id,'direct')]['input_tokens']}
                    for c in self.proposal.cases],
                'rows':rows}


def save_private(root, name, result):
    """Fresh owner-only JSON under this repository's ignored data/; never stdout."""
    import subprocess
    root=Path(root).resolve();data=root/'data'
    require(data.resolve()==data,'Refuse redirected data directory')
    require(Path(name).name==name and name.endswith('.json'),'Use a simple JSON filename')
    destination=data/name
    check=subprocess.run(['git','check-ignore','--quiet',str(destination)],cwd=root,capture_output=True)
    require(check.returncode==0,'Destination must be Git-ignored')
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False).encode()
    fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream: stream.write(payload)
