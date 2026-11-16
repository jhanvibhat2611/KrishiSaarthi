"""Preparation-only records; no retriever, tokenizer, decoder or reference loader."""
from typing import Literal
from pydantic import Field, model_validator
from .qa_comparison_fake import (
    StrictModel, QAC1Proposal, SyntheticCase, FakeFailure, run_fake_pair,
    proposal_manifest, fingerprint,
)


class E5ComparisonProposal(QAC1Proposal):
    proposal_version: Literal['qa-readiness-e5-v1'] = 'qa-readiness-e5-v1'
    selected_retriever: Literal['e5'] = 'e5'
    retrieval_model_id: Literal['intfloat/multilingual-e5-base']
    retrieval_revision: Literal['d128750597153bb5987e10b1c3493a34e5a4502a']
    retrieval_tokenizer_revision: Literal['d128750597153bb5987e10b1c3493a34e5a4502a']
    embedding_cache_key: str = Field(pattern=r'^[a-f0-9]{64}$')
    embedding_manifest_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    retrieval_model_fingerprint: str = Field(pattern=r'^[a-f0-9]{64}$')
    generation_packages: dict[str, str]
    chat_template: Literal['pinned tokenizer apply_chat_template; tokenize=True; add_generation_prompt=True']
    prompt_prefill_step_size: Literal[256] = 256

    @model_validator(mode='after')
    def binding(self):
        if self.index_identifier != 'dense-cosine-' + self.retrieval_model_fingerprint:
            raise ValueError('E5 index identifier must match the model fingerprint')
        if self.generation_packages != {'mlx-lm':'0.26.3', 'mlx':'0.26.5', 'transformers':'4.57.1'}:
            raise ValueError('Preserve pinned generator package versions')
        return self


def readiness_manifest(proposal):
    proposal = E5ComparisonProposal.model_validate(proposal.model_dump(), strict=True)
    # Reuse existing manifest accounting through its compatible base fields.
    base = {k:v for k,v in proposal.model_dump().items() if k in QAC1Proposal.model_fields}
    base['proposal_version'] = 'qa-c1-proposed-v1'
    result = proposal_manifest(QAC1Proposal.model_validate(base, strict=True))
    result['configuration'] = proposal.model_dump()
    result['proposal_sha256'] = fingerprint(proposal.model_dump())
    result['input_budget_policy'] = {
        'context_limit_per_arm':8192, 'reserved_output_tokens_per_arm':384,
        'maximum_prompt_tokens_per_arm':7808,
        'measurement':'Pinned Qwen chat-template tokens, including system messages and special tokens',
        'equal_input_tokens_claimed':False,
        'overflow':'Block pair before generation; no truncation, passage dropping or adaptive Top-K',
        'record':'Both prompt token counts and rag-minus-direct difference; retrieval/load time separately',
    }
    result['future_measurement_fields'] += ['finish_reason', 'partial_raw_output', 'prompt_tokens_sha256',
                                          'retrieval_latency_seconds', 'load_seconds', 'peak_memory_bytes']
    return result


class SimulatedTelemetry(StrictModel):
    """Supplied test values, never measurements of model execution."""
    provenance: Literal['SIMULATED_NOT_MODEL_MEASUREMENTS'] = 'SIMULATED_NOT_MODEL_MEASUREMENTS'
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0, le=384)
    runtime_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    finish_reason: Literal['stop', 'length', 'repetition', 'runtime_error'] | None = None


def record_synthetic_pair(case: SyntheticCase, direct: str | FakeFailure, rag: str | FakeFailure,
                          direct_telemetry: SimulatedTelemetry, rag_telemetry: SimulatedTelemetry):
    """Record supplied outputs and test telemetry. Not a backend integration hook."""
    telemetry = [SimulatedTelemetry.model_validate(t.model_dump(), strict=True)
                 for t in (direct_telemetry, rag_telemetry)]
    for value, t in zip((direct, rag), telemetry):
        if t.input_tokens is not None and t.input_tokens + 384 > 8192:
            raise ValueError('Prompt budget exceeded; no truncation or unequal output cap')
        expected = {'output_length_exhaustion':'length', 'repetition':'repetition',
                    'model_runtime_error':'runtime_error'}.get(value.category) if isinstance(value, FakeFailure) else 'stop'
        if t.finish_reason is not None and t.finish_reason != expected:
            raise ValueError('Finish reason conflicts with supplied output/failure')
    result = run_fake_pair(case, direct_output=direct, rag_output=rag)
    # Legacy fake demo retains its old 256-token placeholder; this new artifact
    # explicitly uses the 384-token readiness proposal, never rewrites old outputs.
    result['schema_version'] = 3
    result['planned_generation_budget']['max_output_tokens_per_arm'] = 384
    for row, t in zip(result['results'], telemetry):
        row['simulated_telemetry'] = t.model_dump()
        # Existing measured fields remain null; fake numbers cannot masquerade as measurements.
    a, b = telemetry
    result['input_budget_comparison'] = {
        'provenance':'SIMULATED_NOT_MODEL_MEASUREMENTS',
        'direct_prompt_tokens':a.input_tokens, 'rag_prompt_tokens':b.input_tokens,
        'rag_minus_direct_tokens':None if a.input_tokens is None or b.input_tokens is None else b.input_tokens-a.input_tokens,
        'maximum_prompt_tokens_per_arm':7808, 'equal_total_compute_claimed':False,
    }
    return result
