"""Regenerate SYNTHETIC development fixtures; never research annotations."""
from pathlib import Path
from reportlab.pdfgen import canvas
from krishisaarthi.ingestion.plaintext import PlainTextIngestor
from krishisaarthi.schemas import SourceMetadata, Query, EvidenceSpan, GoldLabel
from krishisaarthi.ingestion.storage import write_jsonl

ROOT = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures'
TEXT = '''SYNTHETIC DEVELOPMENT FIXTURE - NOT AGRICULTURAL ADVICE OR RESEARCH EVIDENCE.
# Seed record
The imaginary wheat seed record contains the marker seedmarker.
# Water record
The imaginary wheat water record contains the marker watermarker.
# Storage record
The imaginary wheat storage record contains the marker storemarker.
'''
ROOT.joinpath('synthetic_wheat.txt').write_bytes(TEXT.encode('utf-8'))
pdf = canvas.Canvas(str(ROOT / 'synthetic_wheat.pdf'), invariant=1, pageCompression=0)
for page, (heading, sentence) in enumerate([
    ('# Inspection record', 'The imaginary wheat inspection record contains inspectmarker.'),
    ('# Harvest record', 'The imaginary wheat harvest record contains harvestmarker.')], 1):
    pdf.setFont('Helvetica-Bold', 11)
    pdf.drawString(48, 790, 'SYNTHETIC DEVELOPMENT FIXTURE - NOT RESEARCH EVIDENCE')
    pdf.setFont('Helvetica', 11)
    pdf.drawString(48, 755, heading)
    pdf.drawString(48, 730, sentence)
    pdf.drawString(48, 60, f'Page {page}. Invented test text. Not agricultural advice.')
    pdf.showPage()
pdf.save()
doc = PlainTextIngestor().ingest(ROOT / 'synthetic_wheat.txt', SourceMetadata(document_id='synthetic-text'))
queries = [Query(query_id='fixture-en', text='seedmarker', language='en'),
           Query(query_id='fixture-hi', text='गेहूँ के पानी वाले रिकॉर्ड में क्या है?', language='hi'),
           Query(query_id='fixture-hinglish', text='wheat ka storemarker record', language='hinglish'),
           Query(query_id='fixture-unanswerable', text='What is the imaginary dragon yield?', language='en')]
gold = []
for query, phrase in zip(queries[:3], [
    'The imaginary wheat seed record contains the marker seedmarker.',
    'The imaginary wheat water record contains the marker watermarker.',
    'The imaginary wheat storage record contains the marker storemarker.']):
    start = doc.canonical_text.index(phrase)
    span = EvidenceSpan(document_id=doc.document_id, content_sha256=doc.content_sha256,
                        canonical_sha256=doc.canonical_sha256, start_char=start, end_char=start + len(phrase))
    gold.append(GoldLabel(query_id=query.query_id, answerable=True, evidence_spans=[span]))
gold.append(GoldLabel(query_id=queries[3].query_id, answerable=False))
write_jsonl(ROOT / 'synthetic_queries.jsonl', queries)
write_jsonl(ROOT / 'synthetic_gold.jsonl', gold)
