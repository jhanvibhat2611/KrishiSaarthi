"""Fake-only synthetic comparison. No model, input-dataset or real-mode option."""
import argparse
import json
from pathlib import Path
from krishisaarthi.evaluation.qa_comparison_fake import (
    CaseIdentity, QAC1Proposal, blank_scoring_template, proposal_manifest, synthetic_demo,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional fresh directory inside local data/; otherwise print only")
    parser.add_argument("--prepare-qa-c1", action="store_true",
                        help="Write an UNAPPROVED proposal and empty 24-slot scoring form; no inference/retrieval")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    proposal = None
    if args.prepare_qa_c1:
        # This reads only declarative settings/IDs, never the paths named inside them.
        proposal = QAC1Proposal.model_validate_json((root / "configs/qa-c1.proposed.json").read_text(), strict=True)
    result = synthetic_demo()
    if args.output is None:
        print(json.dumps({"proposal": proposal_manifest(proposal), "synthetic_demo": result}
                         if proposal else result, ensure_ascii=False, indent=2))
        return
    output = args.output.resolve()
    if not output.is_relative_to(root / "data"):
        parser.error("Output must remain inside the project's ignored data/ directory")
    output.mkdir(parents=True, exist_ok=False)
    (output / "fake-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    cases = proposal.cases if proposal else [CaseIdentity(case_id=result["paired_identity"]["case_id"],
        group_id=result["paired_identity"]["case_id"], language=result["paired_identity"]["language"])]
    form = blank_scoring_template(cases)
    (output / "human-scoring.pending.json").write_text(json.dumps(form, indent=2) + "\n")
    if proposal:
        (output / "qa-c1.proposed-manifest.json").write_text(json.dumps(proposal_manifest(proposal), indent=2) + "\n")
        (output / "qa-c1.proposal.schema.json").write_text(json.dumps(QAC1Proposal.model_json_schema(), indent=2) + "\n")
    print(f"Fake-only artifacts: {output}; real model calls: 0; human scores: unfilled")


if __name__ == "__main__":
    main()
