"""Synthetic strings only: never load a model or read a research dataset."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError
from krishisaarthi.evaluation.qa_comparison_fake import (
    FakeFailure, Question, SyntheticCase, SyntheticEvidence, build_direct_prompt,
    run_fake_pair, synthetic_demo, CaseIdentity, FakeAttempt, QAC1Proposal,
    account_fake_attempts, blank_scoring_template, proposal_manifest, COMMON,
)


def case():
    return SyntheticCase(case_id="synthetic-test", question=Question(text="What fictional tags?", language="en"),
        retrieved_evidence=[SyntheticEvidence(passage_id="synthetic-p1", document_id="demo-d1",
                                              text="SYNTHETIC: EVIDENCE_MARKER_917; one tag is Aspen."),
                            SyntheticEvidence(passage_id="synthetic-p2", document_id="demo-d1",
                                              text="SYNTHETIC: A second tag is Elm.")])


def draft(rag=False):
    claim = {"text": "A fictional tag is Aspen."}
    if rag:
        claim["evidence_handles"] = ["E1"]
    return json.dumps({"status": "draft", "claims": [claim], "reason": "Unverified explanation."})


def pair(direct=None, rag=None):
    return run_fake_pair(case(), direct_output=draft() if direct is None else direct,
                         rag_output=draft(True) if rag is None else rag)


def test_direct_receives_query_only_and_rag_receives_evidence():
    result = pair()
    direct, rag = result["results"]
    assert json.loads(direct["prompt"][1]["content"]) == case().question.model_dump()
    encoded = json.dumps(direct["prompt"])
    for forbidden in ("EVIDENCE_MARKER_917", "synthetic-p1", "demo-d1", "Aspen", '"E1"'):
        assert forbidden not in encoded
    assert "EVIDENCE_MARKER_917" in json.dumps(rag["prompt"])
    assert direct["case_id"] == rag["case_id"] == result["paired_identity"]["case_id"]
    assert direct["question_sha256"] == rag["question_sha256"]
    assert result["real_model_calls"] == 0 and result["model"] is None


@pytest.mark.parametrize("field", ["reference_answer", "gold", "answerable", "evidence_ids", "reviewer_decisions"])
def test_reference_and_label_fields_rejected_before_prompt_building(field):
    payload = case().model_dump()
    payload[field] = "SECRET_REFERENCE_MARKER"
    with pytest.raises(ValidationError):
        SyntheticCase.model_validate(payload)
    payload = case().question.model_dump()
    payload[field] = "SECRET_REFERENCE_MARKER"
    with pytest.raises(ValidationError):
        Question.model_validate(payload)
    payload = case().retrieved_evidence[0].model_dump()
    payload[field] = "SECRET_REFERENCE_MARKER"
    with pytest.raises(ValidationError):
        SyntheticEvidence.model_validate(payload)


def test_no_reference_or_backend_argument_and_revalidate_constructed_input():
    with pytest.raises(TypeError):
        run_fake_pair(case(), direct_output=draft(), rag_output=draft(True), reference_answer="secret")
    with pytest.raises(TypeError):
        run_fake_pair(case(), direct_output=lambda: None, rag_output=draft(True))
    bad = case().model_copy(update={"case_id": "real-corpus"})
    with pytest.raises(ValidationError):
        run_fake_pair(bad, direct_output=draft(), rag_output=draft(True))


def test_no_fake_scores_or_human_approval_and_direct_citations_na():
    direct, rag = pair()["results"]
    for row in (direct, rag):
        assert row["status"] == "review_required"
        assert row["confidence"] == "not_estimated"
        assert row["reason_role"] == "unverified_metadata_not_answer"
        assert "explanation" not in row["answer"]
        assert all(v is None for k, v in row["human_review"].items() if k != "review_status")
        assert row["generation_runtime_seconds"] is None and row["output_tokens"] is None
    assert direct["citation_diagnostics"]["status"] == "not_applicable"
    assert rag["citation_diagnostics"]["status"] == "resolved_structure_only"
    assert not rag["citation_diagnostics"]["entailment_verified"]
    ref = rag["citation_diagnostics"]["resolved_references"][0]
    assert ref["passage_id"] == "synthetic-p1" and ref["document_id"] == "demo-d1"
    assert ref["start_char"] == 0 and ref["end_char"] == len(case().retrieved_evidence[0].text)


@pytest.mark.parametrize("raw,category", [
    ('{"status":', "invalid_json"),
    ('{"status":"uncertain","status":"uncertain","claims":[],"reason":"x"}', "invalid_json"),
    ('{"status":"draft","claims":[{"text":"x","text":"x"}],"reason":""}', "invalid_json"),
    ('{"status":"draft","claims":[],"reason":NaN}', "invalid_json"),
    ('{"status":"approved","claims":[],"reason":"x"}', "invalid_status"),
    ('{}', "schema_failure"),
    ('{"status":"draft","claims":[],"reason":""}', "incorrect_status"),
    ('{"status":"insufficient","claims":[{"text":"invented"}],"reason":"missing"}', "incorrect_status"),
    ('{"status":"uncertain","claims":[],"reason":" "}', "incorrect_status"),
])
def test_raw_failures_preserved_not_counted_as_abstention(raw, category):
    row = pair(direct=raw)["results"][0]
    assert row["raw_output"] == raw
    assert row["failure_category"] == category and row["status"] == "error"
    assert row["answer"] is None and row["claims"] == []
    assert row["retries"] == row["repairs"] == 0
    assert row["human_review"]["abstention_appropriateness"] is None


@pytest.mark.parametrize("category", ["model_runtime_error", "output_length_exhaustion", "repetition"])
def test_simulated_backend_failure_preserves_partial_raw(category):
    for row in pair(direct=FakeFailure(category=category, raw_output="partial"),
                    rag=FakeFailure(category=category, raw_output="partial"))["results"]:
        assert row["raw_output"] == "partial" and row["failure_category"] == category
        assert row["status"] == "error" and row["simulated_failure"]


@pytest.mark.parametrize("handles", [["E99"], ["E1", "E1"], [" E1"], ["e1"]])
def test_rag_citation_validation_is_separate(handles):
    raw = json.dumps({"status": "draft", "claims": [{"text": "Fake claim", "evidence_handles": handles}], "reason": ""})
    direct, rag = pair(rag=raw)["results"]
    assert direct["status"] == "review_required"
    assert direct["citation_diagnostics"]["status"] == "not_applicable"
    assert rag["status"] == "error" and rag["failure_category"] == "invalid_citation"
    assert not rag["citation_diagnostics"]["resolved_references"]


def test_direct_has_no_handle_requirement_but_cannot_smuggle_extra_schema_fields():
    assert pair()["results"][0]["status"] == "review_required"
    assert pair(direct=draft(True))["results"][0]["failure_category"] == "schema_failure"
    assert pair(rag=draft())["results"][1]["failure_category"] == "schema_failure"


@pytest.mark.parametrize("status", ["insufficient", "uncertain"])
def test_abstention_not_certified_as_correct(status):
    raw = json.dumps({"status": status, "claims": [], "reason": "Canned missing-context explanation"})
    for row in pair(direct=raw, rag=raw)["results"]:
        assert row["status"] == status and row["answer"] is None
        assert row["human_review"]["abstention_appropriateness"] is None


def test_multi_passage_fixture_and_empty_evidence_do_not_invent_support():
    r = synthetic_demo()
    assert len(r["results"][1]["citation_diagnostics"]["resolved_references"]) == 2
    assert r["results"][1]["human_review"]["completeness"] is None
    empty = case().model_copy(update={"retrieved_evidence": []})
    row = run_fake_pair(empty, direct_output=draft(), rag_output=draft(True))["results"][1]
    assert row["failure_category"] == "invalid_citation"
    assert build_direct_prompt(empty.question) == build_direct_prompt(case().question)


def test_fixed_metadata_and_deterministic_fake_pair():
    assert pair() == pair()
    budget = pair()["planned_generation_budget"]
    assert budget["max_output_tokens_per_arm"] == 256
    assert budget["calls_per_arm"] == 1 and budget["repairs"] == 0
    assert set(pair()["prompt_versions"]) == {"direct", "rag"}


def test_cli_default_is_fake_and_real_flags_are_unavailable():
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts/run_fake_qa_comparison.py")]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["real_model_calls"] == 0
    bad = subprocess.run(cmd + ["--model-dir", "not-a-model"], cwd=root, env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "unrecognized arguments" in bad.stderr


def proposal():
    path = Path(__file__).resolve().parents[1] / "configs/qa-c1.proposed.json"
    return QAC1Proposal.model_validate_json(path.read_text(), strict=True)


def test_qac1_proposal_is_unapproved_fixed_budget_and_metadata_only():
    p = proposal()
    result = proposal_manifest(p)
    assert p.status == "PROPOSED_UNAPPROVED" and p.real_execution_authorized is False
    assert p.source_use_authorized_by_proposal is False and p.selected_retriever is None
    assert len(p.cases) == 12 and len({c.group_id for c in p.cases}) == 4
    assert p.max_calls_per_arm == 12 and p.max_calls_total == 24
    assert p.max_output_tokens_per_call == 384 and result["maximum_proposed_generated_tokens"] == 9216
    assert p.generation_calls_per_question_arm == 1 and p.repairs == p.retries == 0
    assert p.model_revision == p.tokenizer_revision
    assert result["real_model_calls"] == 0
    assert result["planned_accounting_no_execution"]["overall"]["not_attempted_slots"] == 24
    assert result["planned_accounting_no_execution"]["overall"]["simulated_attempts"] == 0
    assert all(result[k] is None for k in ("reference_approval", "architecture_approval", "execution_approval"))
    # Even nonexistent paths remain declarative: manifest creation never opens them.
    changed = p.model_copy(update={"mlx_config_path": "/missing/model.json", "index_manifest_path": "/missing/index.json"})
    assert proposal_manifest(changed)["configuration_paths_are_metadata_only"]


@pytest.mark.parametrize("update", [
    {"real_execution_authorized": True}, {"status": "APPROVED"},
    {"source_use_authorized_by_proposal": True}, {"max_calls_total": 25},
    {"max_output_tokens_per_call": 768}, {"generation_calls_per_question_arm": 2},
    {"retries": 1}, {"reference_answer": "SECRET"}, {"gold": "SECRET"},
    {"selected_retriever": "new_method"},
])
def test_proposal_rejects_execution_flags_unapproved_budgets_and_labels(update):
    payload = proposal().model_dump()
    payload.update(update)
    with pytest.raises(ValidationError):
        QAC1Proposal.model_validate(payload, strict=True)


def test_proposal_rejects_duplicate_or_wrong_case_matrix():
    payload = proposal().model_dump()
    payload["cases"][0] = payload["cases"][1]
    with pytest.raises(ValidationError):
        QAC1Proposal.model_validate(payload, strict=True)
    payload = proposal().model_dump()
    payload["cases"][0]["answerability"] = "secret"
    with pytest.raises(ValidationError):
        QAC1Proposal.model_validate(payload, strict=True)


def accounting_fixture():
    cases = [CaseIdentity(case_id="synthetic-" + lang, group_id="synthetic-group", language=lang)
             for lang in ("en", "hi", "hinglish")]
    def obs(lang, arm, outcome, error=None, completed=True, valid=True):
        return FakeAttempt(case_id="synthetic-" + lang, language=lang, arm=arm, attempted=True,
                           response_completed=completed, structured_valid=valid,
                           outcome=outcome, failure_category=error)
    observations = [obs("en", "direct", "error", "invalid_json", valid=False),
                    obs("en", "rag", "answer"),
                    obs("hi", "direct", "error", "model_runtime_error", completed=False, valid=False),
                    obs("hi", "rag", "error", "invalid_citation"),
                    obs("hinglish", "direct", "insufficient")]
    return cases, observations


def test_failures_and_unattempted_slots_stay_in_primary_denominators():
    cases, observations = accounting_fixture()
    result = account_fake_attempts(cases, observations)
    total = result["overall"]
    assert total["planned_question_arm_slots"] == total["primary_completion_denominator"] == 6
    assert total["simulated_attempts"] == 5 and total["not_attempted_slots"] == 1
    assert total["simulated_completed_responses"] == 4
    assert total["simulated_structured_valid_responses"] == 3
    assert total["substantive_answer_drafts"] == 1 and total["deliberate_withholding"] == 1
    assert total["failure_suppressed_answers"] == 3
    assert total["simulated_completion_rate_over_planned"] == 4 / 6
    assert total["failure_categories"] == {"invalid_json": 1, "model_runtime_error": 1, "invalid_citation": 1}
    assert total["actual_model_attempts"] == total["actual_completed_model_responses"] == 0
    assert total["measured_prompt_tokens"] is total["measured_generated_tokens"] is None
    assert result["by_arm"]["direct"]["primary_completion_denominator"] == 3
    assert result["by_arm"]["rag"]["primary_completion_denominator"] == 3
    assert result["paired_completion_denominator"] == 3 and result["both_completed_pairs"] == 1
    assert result["paired_cases"][2]["arms"]["rag"]["outcome"] == "not_attempted"


def test_accounting_rejects_duplicate_unplanned_or_inconsistent_observations():
    cases, observations = accounting_fixture()
    with pytest.raises(ValueError):
        account_fake_attempts(cases + [cases[0]], observations)
    with pytest.raises(ValueError):
        account_fake_attempts(cases, observations + [observations[0]])
    with pytest.raises(ValueError):
        account_fake_attempts(cases, [observations[0].model_copy(update={"case_id": "unknown"})])
    with pytest.raises(ValidationError):
        account_fake_attempts(cases, [observations[0].model_copy(update={"outcome": "insufficient"})])
    with pytest.raises(ValidationError):
        FakeAttempt(case_id="x", language="en", arm="direct", attempted=False,
                    response_completed=True, structured_valid=True, outcome="answer")


@pytest.mark.parametrize("category", ["output_length_exhaustion", "repetition", "model_runtime_error"])
def test_incomplete_simulated_generations_never_become_completed_abstentions(category):
    result = pair(direct=FakeFailure(category=category, raw_output="partial"))
    count = result["synthetic_attempt_accounting"]["by_arm"]["direct"]
    assert count["primary_completion_denominator"] == count["simulated_attempts"] == 1
    assert count["simulated_completed_responses"] == count["deliberate_withholding"] == 0
    assert count["failure_suppressed_answers"] == 1


def test_qac1_rubric_fields_empty_and_direct_na_is_not_a_human_judgment():
    template = blank_scoring_template(proposal().cases)
    assert len(template["rows"]) == 24 and template["reference_scoring_enabled"] is False
    assert "unsupported_by_authorized_corpus" in template["rubric_options_not_judgments"]["corpus_grounded_support"]
    for row in template["rows"]:
        assert all(v is None for k, v in row["human_review"].items() if k != "review_status")
        assert row["rag_human_citation_structure"] is row["rag_human_claim_to_evidence_support"] is None
        assert row["supplied_evidence_citation_applicability"] == ("not_applicable" if row["arm"] == "direct" else "applicable")


def test_shared_general_instructions_and_manifest_do_not_enter_prompts():
    result = pair()
    for row in result["results"]:
        assert row["prompt"][0]["content"].startswith(COMMON)
        assert "pilot-v01-in-06" not in json.dumps(row["prompt"])
        assert "reference_approval" not in json.dumps(row["prompt"])
    assert result["paired_identity"]["language"] == "en"


def test_prepare_cli_is_metadata_and_fake_only_without_model_imports():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    code = ("import runpy,sys; sys.argv=['run_fake_qa_comparison.py','--prepare-qa-c1']; "
            "runpy.run_path('scripts/run_fake_qa_comparison.py',run_name='__main__'); "
            "assert not any(n.split('.')[0] in {'mlx','mlx_lm','torch','transformers'} for n in sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert result["proposal"]["status"] == "PROPOSED_UNAPPROVED_NOT_A_FROZEN_PROTOCOL"
    assert result["synthetic_demo"]["real_model_calls"] == 0
