import hashlib
import json
from pathlib import Path
from ..config import text_hash
from ..schemas import Document, PageReference

PAGE_SEPARATOR = "\n\f\n"


def make_document(path, metadata, pages, media_type, extractor, version):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    refs, parts, offset = [], [], 0
    for number, text in pages:
        if parts:
            offset += len(PAGE_SEPARATOR)
        refs.append(PageReference(page_number=number, canonical_start=offset, canonical_end=offset + len(text)))
        parts.append(text)
        offset += len(text)
    canonical = PAGE_SEPARATOR.join(parts)
    return Document(document_id=metadata.document_id or "doc-" + digest,
        content_sha256=digest, canonical_sha256=text_hash(canonical), original_filename=path.name,
        original_file_path=str(path.resolve()), source_url=metadata.source_url,
        media_type=media_type, language=metadata.language, domain_metadata=metadata.domain_metadata,
        canonical_text=canonical, pages=refs,
        extraction_metadata={"extractor": extractor, "version": version, "canonicalization": "verbatim-pages-v1", "page_separator": PAGE_SEPARATOR})


def save_document(document: Document, root: Path) -> Document:
    doc = document.model_copy(deep=True)
    # Hash the ID for safe paths, including externally supplied document IDs.
    key = text_hash(doc.document_id)
    extracted = root / "extracted" / key / doc.canonical_sha256
    original = root / "originals" / key / doc.content_sha256 / doc.original_filename
    payload = Path(doc.original_file_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != doc.content_sha256:
        raise ValueError("Original changed after extraction")
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(payload)
    doc.original_file_path = str(original.resolve())
    extracted.mkdir(parents=True, exist_ok=True)
    (extracted / "canonical.txt").write_bytes(doc.canonical_text.encode("utf-8"))
    for i, page in enumerate(doc.pages, 1):
        target = extracted / f"page-{i:04d}.txt"
        target.write_bytes(doc.canonical_text[page.canonical_start:page.canonical_end].encode("utf-8"))
        page.extracted_text_path = str(target.resolve())
    return doc


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


def read_jsonl(path: Path, model):
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{number}: blank record")
        try:
            records.append(model.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    return records
