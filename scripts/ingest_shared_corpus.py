"""
Bulk-ingests every PDF in data/shared_corpus/ as shared, queryable-by-
everyone documents (as opposed to a specific user's private upload).

Skips any PDF whose doc_id is already indexed, since re-ingesting wastes
scarce free-tier LLM quota re-describing charts that were already
processed. Pass --force to re-ingest everything anyway (e.g. after
changing the parsing/chunking logic).

Run:  python -m scripts.ingest_shared_corpus
      python -m scripts.ingest_shared_corpus --force
"""
import sys
from pathlib import Path
import uuid

from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_raw
from src.rag import vector_store

CORPUS_DIR = Path("data/shared_corpus")


def main():
    force = "--force" in sys.argv
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {CORPUS_DIR}/. Drop some annual reports there first.")
        sys.exit(1)

    for pdf_path in pdfs:
        doc_id = pdf_path.stem or str(uuid.uuid4())[:8]

        if not force and vector_store.doc_exists(doc_id):
            print(f"Skipping {pdf_path.name} (doc_id={doc_id}) — already indexed. Use --force to re-ingest.")
            continue

        print(f"Ingesting {pdf_path.name} as doc_id={doc_id} ...")
        raw = parse_pdf(pdf_path, doc_id=doc_id)
        chunks = chunk_raw(raw)
        count = vector_store.upsert_chunks(chunks, source="shared")
        print(f"  -> {count} chunks indexed "
              f"({sum(1 for c in chunks if c.source_type == 'chart')} from charts, "
              f"{sum(1 for c in chunks if c.source_type == 'table')} from tables)")

    print("Done.")


if __name__ == "__main__":
    main()
