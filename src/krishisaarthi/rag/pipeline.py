import time
from .models import GeneratedAnswer, RAGResponse
from .evidence import EvidenceSufficiencyChecker, CitationValidator
from .generation import PromptBuilder
from ..validation import require

class RAGPipeline:
    def __init__(self,selector,generator,checker=None,prompt_builder=None,validator=None):
        self.selector=selector;self.generator=generator
        self.checker=checker or EvidenceSufficiencyChecker()
        self.prompt_builder=prompt_builder or PromptBuilder()
        self.validator=validator or CitationValidator()

    def run(self,question,k=5,review=None,mode="conservative"):
        require(mode in {"conservative","research_draft"}, "Unknown RAG execution mode")
        started=time.perf_counter();evidence=self.selector.select(question,k)
        sources=[]
        for p in evidence.passages:
            d=self.selector.documents[p.document_id]
            sources.append({'passage_id':p.passage_id,'document_id':p.document_id,
                'original_filename':d.original_filename,'source_url':d.source_url,
                'evidence_span':p.evidence_span.model_dump(),'page_references':[r.model_dump() for r in p.page_references],
                'document_metadata':d.domain_metadata,'passage_metadata':p.domain_metadata})
        response=RAGResponse(original_question=question.text,retrieved_passage_ids=[p.passage_id for p in evidence.passages],
            evidence_fingerprint=evidence.fingerprint,status='insufficient_evidence',answerability='not_established',sources=sources,
            execution_mode=mode,
            warnings=['Source scope, dates and geographic/cultivation restrictions remain applicable; current agricultural correctness is not established.',
                'Citation validity does not establish factual entailment. Unsupported claims require review, not automatic certification.'])
        assessment=self.checker.assess(evidence,review);response.warnings.extend(assessment.reasons)
        if not assessment.sufficient_to_attempt and (mode=="conservative" or not evidence.passages):
            response.diagnostics={'runtime_seconds':time.perf_counter()-started,'generator_called':False};return response
        if mode=='conservative':
            response.answerability='sufficient_to_attempt'
        else:
            response.warnings.append('RESEARCH DRAFT: semantic sufficiency is NOT established; candidate generation is permitted for evaluation only. Missing support must be reported.')
        try:
            result=GeneratedAnswer.model_validate(self.generator.generate(self.prompt_builder.build(evidence),evidence))
        except Exception as exc:
            response.status='generator_error';response.warnings.append('Generator failed: '+type(exc).__name__)
        else:
            errors=self.validator.validate(result.claims,evidence)
            if errors:
                response.status='invalid_output';response.warnings.extend(errors)
            elif result.abstain or result.missing_context or not result.claims:
                response.status='insufficient_evidence';response.answerability='not_established'
                response.warnings.extend(result.missing_context or ['Generator provided no supported draft'])
            else:
                response.status='review_required';response.claims=result.claims
                response.generated_answer='\n'.join(c.text for c in result.claims)
                response.warnings.append('All generated claims are UNVERIFIED, including those with structurally valid source quotes; review factual support before use.')
        response.diagnostics={'runtime_seconds':time.perf_counter()-started,'generator_called':True,
            'generator':getattr(self.generator,'last_metrics',{})}
        return response
