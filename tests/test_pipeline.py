import json
import subprocess
import sys
from pathlib import Path
from conftest import FIXTURES

ROOT = Path(__file__).resolve().parents[1]


def test_cli_end_to_end(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, '-m', 'krishisaarthi.cli', '--config', str(ROOT / 'configs/baseline.toml'), *map(str, args)], cwd=ROOT, text=True, capture_output=True, check=True)
    run('ingest', '--manifest', FIXTURES / 'manifest.jsonl', '--output-dir', tmp_path)
    docs, passages = tmp_path / 'documents.jsonl', tmp_path / 'passages.jsonl'
    run('chunk', '--documents', docs, '--output', passages)
    run('search', '--passages', passages, '--query', 'seedmarker', '--k', '2')
    output = tmp_path / 'report.json'
    args = ('evaluate', '--documents', docs, '--passages', passages, '--queries', FIXTURES / 'synthetic_queries.jsonl', '--gold', FIXTURES / 'synthetic_gold.jsonl', '--output', output)
    run(*args)
    report = json.loads(output.read_text())
    assert report['evaluated_queries'] == 3 and report['excluded_queries'] == 1
    assert report['coverage_rule'] == 'strict_union'
    assert report['warnings'] == []
    assert report['manifest']['config']['chunking']['size'] == 256
    assert report['manifest']['dependencies']['rank-bm25'] == '0.2.2'
    run(*args)
    second = json.loads(output.read_text())
    assert report['run_id'] == second['run_id']
    assert report['metrics'] == second['metrics']
