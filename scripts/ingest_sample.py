"""Build a Chroma collection from the sample Markdown docs.

Usage:
    python scripts/ingest_sample.py [--source data/sample_docs]
                                    [--collection my-docs]
                                    [--persist-dir ./chroma_db]

Uses the Project 1-compatible schema (chunk ids, metadata, cosine space),
so this collection is interchangeable with Project 1's output.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedder import SentenceTransformerEmbedder  # noqa: E402
from app.ingest import ingest_folder  # noqa: E402
from app.retrieval import load_collection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sample docs into Chroma.")
    parser.add_argument("--source", default="data/sample_docs")
    parser.add_argument("--collection", default="my-docs")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder(args.model)
    collection = load_collection(args.persist_dir, args.collection)
    report = ingest_folder(
        Path(args.source), collection, embedder.embed_query,
        chunk_size=args.chunk_size, overlap=args.overlap,
    )
    print(f"Documents: {report['documents']}")
    print(f"Pages: {report['pages']}")
    print(f"Chunks produced: {report['chunks_produced']}")
    print(f"Chunks indexed: {report['chunks_indexed']}")
    print(f"Duplicates dropped: {report['duplicates_dropped']}")


if __name__ == "__main__":
    main()
