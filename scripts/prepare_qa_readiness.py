"""Synthetic preparation only. No dataset, inference or retrieval CLI options."""
import argparse
import json
from pathlib import Path
from krishisaarthi.evaluation.qa_readiness import (
    E5ComparisonProposal, readiness_manifest, SimulatedTelemetry, record_synthetic_pair,
)
from krishisaarthi.evaluation.qa_comparison_fake import (
    Question, SyntheticCase, SyntheticEvidence, FakeFailure, blank_scoring_template,
)


def prepare():
    root = Path(__file__).resolve().parents[1]
    proposal = E5ComparisonProposal.model_validate_json(
        (root/'configs/qa-readiness-e5.proposed.json').read_text(), strict=True)
    examples = []
    for language, question in [('en','Which fictional tag is used?'),
                               ('hi','कौन-सा काल्पनिक टैग इस्तेमाल हुआ है?'),
                               ('hinglish','Kaunsa fictional tag use hua hai?')]:
        case = SyntheticCase(case_id='synthetic-readiness-'+language,
            question=Question(text=question, language=language), retrieved_evidence=[
                SyntheticEvidence(passage_id='synthetic-tag', document_id='demo-notebook',
                                  text='SYNTHETIC: The fictional tag is Aspen.')])
        # Invented plumbing examples, deliberately not answers to agricultural questions.
        direct = json.dumps({'status':'uncertain','claims':[],'reason':'Canned uncertainty'})
        rag = json.dumps({'status':'draft','claims':[{'text':'Aspen','evidence_handles':['E1']}],'reason':'Canned fixture'})
        if language == 'hi': direct = FakeFailure(category='output_length_exhaustion', raw_output='{"status":')
        if language == 'hinglish': rag = '{invalid fixture JSON'
        examples.append(record_synthetic_pair(case,direct,rag,
            SimulatedTelemetry(input_tokens=80,output_tokens=384 if language=='hi' else 12,runtime_seconds=0.2,
                               finish_reason='length' if language=='hi' else 'stop'),
            SimulatedTelemetry(input_tokens=125,output_tokens=18,runtime_seconds=0.3,finish_reason='stop')))
    return {'proposal':readiness_manifest(proposal), 'synthetic_examples':examples,
            'human_review':blank_scoring_template(proposal.cases)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Fresh Git-ignored data directory')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    if not output.is_relative_to(root/'data'): parser.error('Output must stay in ignored data/')
    payload=prepare()
    output.mkdir(parents=True,exist_ok=False)
    for name, value in payload.items():
        (output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print('Preparation saved; model calls=0, retrieval calls=0, human scores=unfilled.')


if __name__=='__main__': main()
