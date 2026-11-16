"""QA-C3: metadata preflight and synthetic private-process rehearsal only."""
import argparse
import json
import sys
from pathlib import Path
from krishisaarthi.evaluation.qa_runtime import check_local,dry_run,consume_synthetic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('check')
    dry=sub.add_parser('dry-run');dry.add_argument('--directory',type=Path,required=True)
    consume=sub.add_parser('consume-synthetic');consume.add_argument('--directory',type=Path,required=True);consume.add_argument('--handoff-sha256',required=True)
    real=sub.add_parser('real',help='Single explicitly approved bounded execution; no retries')
    real.add_argument('--directory',type=Path)
    real.add_argument('--approval-file',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.mode=='real':
        if args.directory is None or args.approval_file is None:parser.error('Real execution disabled without the exact bounded approval and fresh output directory.')
        from krishisaarthi.evaluation.qa_real import execute
        code=execute(root,args.directory,args.approval_file)
        print('Bounded run completed.' if code==0 else 'Bounded run stopped; inspect protected halt record. No retry permitted.')
        raise SystemExit(code)
    try:
        result=check_local(root) if args.mode=='check' else dry_run(root,args.directory) if args.mode=='dry-run' else consume_synthetic(root,args.directory,args.handoff_sha256)
    except Exception:
        parser.exit(1,'Readiness validation failed; no source text or exception payload logged.\n')
    if any(n.split('.')[0] in {'mlx','mlx_lm','torch','transformers'} for n in sys.modules):
        parser.exit(1,'Forbidden model library import in preparation mode.\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
