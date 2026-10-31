"""Empty annotation forms only. This module does not infer or export gold labels."""
import csv
from pathlib import Path

QUESTION_FIELDS = ['query_id', 'question_group_id', 'language', 'question_text',
    'topic', 'region_context', 'observation_period', 'corpus_fingerprint',
    'judgment_pending_answerable_unanswerable', 'judgment_rationale',
    'annotator', 'reviewer', 'review_status', 'equivalence_verified',
    'privacy_verified', 'terminology_notes']
EVIDENCE_FIELDS = ['query_id', 'evidence_id', 'document_id', 'content_sha256',
    'canonical_sha256', 'region_id', 'physical_page', 'printed_page_label',
    'start_char', 'end_char', 'exact_quote_local_only', 'evidence_reviewer',
    'evidence_status']


def write_templates(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    for name, fields in [('questions.csv', QUESTION_FIELDS), ('evidence.csv', EVIDENCE_FIELDS)]:
        with (directory / name).open('x', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerow(fields)
    (directory / 'INSTRUCTIONS.md').write_text('''# Manual pilot annotation — empty templates

There are no generated questions, approved evidence labels or gold records here.
Use language `en`, `hi` or `hinglish`. Start each question judgment and review status
as `pending`. Keep pending rows out of evaluation. A blank row is not unanswerable.

Use one question row per language variant, sharing a question_group_id only when
humans verify equivalent intent. Keep variants together in any future train/test
split. Have a second reader verify meaning, terminology, temporal/location context,
and evidence independently for each language. This pilot is not a complete dataset.

Evidence rows use exact canonical character offsets [start_char, end_char), physical
PDF pages and separately observed printed labels. Preserve hashes. Each evidence
span must be manually verified against the original PDF and covered by the approved
research index; use separate rows for separate pages. Never join disjoint spans.
Do not label a source inconsistency as established fact. A missing survey report is
not evidence that a disease is absent. Unanswerable means unsupported by this frozen
corpus and requires explicit human review and rationale.

Exclude unnecessary personal identifiers from questions and evidence. Exact quotes
are local copyrighted source material; do not publish or commit these filled forms.
Source licenses remain unknown unless independently verified. This workflow does
not grant redistribution permission. No automatic gold export is implemented.
''', encoding='utf-8')
