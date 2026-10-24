import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from .config import load_config, fingerprint
from .schemas import Document, Passage, Query, GoldLabel, SourceMetadata
from .ingestion.pdf import PDFIngestor
from .ingestion.plaintext import PlainTextIngestor
from .ingestion.storage import read_jsonl, write_jsonl, save_document
from .preprocessing.chunking import DeterministicChunker
from .preprocessing.tokenization import TOKENIZER_ID
from .retrieval.bm25 import BM25Retriever
from .evaluation.labels import unique_by, validate_corpus
from .evaluation.runner import evaluate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="KrishiSaarthi Phase 1 retrieval baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--manifest", type=Path, required=True, help="JSONL records: path and optional metadata; paths relative to manifest")
    ingest.add_argument("--output-dir", type=Path, default=Path("data"))
    chunk = sub.add_parser("chunk")
    chunk.add_argument("--documents", type=Path, required=True)
    chunk.add_argument("--output", type=Path, required=True)
    search = sub.add_parser("search")
    search.add_argument("--passages", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--k", type=int, default=5)
    evaluate_parser = sub.add_parser("evaluate")
    for name in ("documents", "passages", "queries", "gold", "output"):
        evaluate_parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "ingest":
            docs = []
            for line in args.manifest.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if not isinstance(row, dict) or not set(row) <= {"path", "metadata"} or "path" not in row:
                    raise ValueError("Manifest rows require path and optional metadata only")
                path = args.manifest.parent / row["path"]
                suffix = path.suffix.lower()
                if suffix not in {".txt", ".pdf"}:
                    raise ValueError("Only .txt and text-based .pdf are supported")
                ingestor = PDFIngestor() if suffix == ".pdf" else PlainTextIngestor()
                docs.append(ingestor.ingest(path, SourceMetadata.model_validate(row.get("metadata", {}))))
            if not docs:
                raise ValueError("Empty ingestion manifest")
            unique_by(docs, "document_id")
            saved = [save_document(doc, args.output_dir) for doc in docs]
            write_jsonl(args.output_dir / "documents.jsonl", saved)
            print(f"Ingested {len(saved)} documents")
        elif args.command == "chunk":
            docs = read_jsonl(args.documents, Document)
            unique_by(docs, "document_id")
            chunker = DeterministicChunker(config.chunking)
            passages = [p for doc in docs for p in chunker.chunk(doc)]
            validate_corpus(docs, passages)
            write_jsonl(args.output, passages)
            print(f"Created {len(passages)} passages; chunking_config_id={chunker.config_id}")
        elif args.command == "search":
            retriever = BM25Retriever(config.bm25)
            retriever.index(read_jsonl(args.passages, Passage))
            for result in retriever.search(Query(query_id="interactive", text=args.query), args.k):
                print(result.model_dump_json())
        else:
            docs = read_jsonl(args.documents, Document)
            passages = read_jsonl(args.passages, Passage)
            report = evaluate(BM25Retriever(config.bm25), read_jsonl(args.queries, Query),
                read_jsonl(args.gold, GoldLabel), docs, passages, config.evaluation)
            identity = {"documents": sorted((d.document_id, d.content_sha256, d.canonical_sha256) for d in docs),
                        "passages": sorted((p.passage_id, p.chunking_config_id, p.evidence_span.model_dump_json()) for p in passages)}
            report["synthetic_documents"] = sum(d.domain_metadata.get("synthetic") is True for d in docs)
            if report["synthetic_documents"]:
                report["development_notice"] = "Contains synthetic development fixtures. Scores are not research findings."
            report["manifest"] = {"config": config.model_dump(), "corpus_fingerprint": fingerprint(identity),
                "chunking_config_ids": sorted({p.chunking_config_id for p in passages}),
                "tokenizer": TOKENIZER_ID, "python": platform.python_version(),
                "dependencies": {name: version(name) for name in ("krishisaarthi", "pydantic", "pypdf", "regex", "rank-bm25", "numpy")},
                "extraction": [{"document_id": d.document_id, **d.extraction_metadata} for d in docs]}
            report["run_id"] = fingerprint({"manifest": report["manifest"], "queries": report["query_fingerprint"], "labels": report["label_fingerprint"]})
            report["created_at"] = datetime.now(timezone.utc).isoformat()
            write_json(args.output, report)
            print(json.dumps({key: report[key] for key in ("metrics", "evaluated_queries", "excluded_queries", "coverage_rule")}, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
