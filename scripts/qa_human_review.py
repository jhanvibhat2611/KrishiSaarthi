"""Prepare / serve private QA-C4 review, without model execution."""
import argparse
from pathlib import Path
from krishisaarthi.evaluation.qa_human_review import prepare, serve

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'serve'])
    parser.add_argument('--directory', default='data/qa-c4-human-review-v1')
    parser.add_argument('--port', type=int, default=8774)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root/args.directory
    if args.action == 'prepare':
        prepare(root, directory)
        print('Prepared private review package; no scores or model calls.')
    else:
        serve(directory, args.port)
