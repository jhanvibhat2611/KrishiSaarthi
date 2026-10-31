"""Real subprocess checks under -O; test assertions themselves are not the guards."""
import ast
import subprocess
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).parents[1]

def test_no_runtime_asserts():
    for p in [*(ROOT/'src').rglob('*.py'),*(ROOT/'scripts').glob('*.py')]:
        assert not any(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(p.read_text()))),p

@pytest.mark.parametrize('optimized',[False,True])
def test_validation_survives_optimization(optimized):
    code='''
from krishisaarthi.validation import require
from krishisaarthi.rag.mlx_generator import validate_checkpoint
for fn in [lambda:require(False,"synthetic failure"),lambda:validate_checkpoint("/nonexistent/synthetic-checkpoint",{})]:
    try:fn()
    except ValueError:pass
    else:raise RuntimeError("Validation was disabled")
print("guards active")
'''
    p=subprocess.run([sys.executable,*(['-O'] if optimized else []),'-c',code],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert 'guards active' in p.stdout

def test_optimized_comparison_rejects_external_output(tmp_path):
    p=subprocess.run([sys.executable,'-O','scripts/compare_dense_pilot.py','--output',str(tmp_path/'outside')],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode!=0 and 'ValueError' in p.stderr
    assert not (tmp_path/'outside').exists()
