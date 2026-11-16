"""Private child process; requires an already consumed bounded authorization marker."""
import sys
from pathlib import Path
from krishisaarthi.evaluation.qa_real import worker
if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in ('resource-probe','producer','consumer'):raise SystemExit(2)
    raise SystemExit(worker(Path(__file__).resolve().parents[1],Path(sys.argv[2]).absolute(),sys.argv[1]))
