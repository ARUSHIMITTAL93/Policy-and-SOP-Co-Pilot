from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from document_pipeline import build_corpus_from_chunks, load_chunks, save_corpus_to_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
INDEX_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "chunk_metadata.pkl"
def validate_api_key() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Set OPENAI_API_KEY in .env before building the index."
        )


def get_client() -> OpenAI:
    load_dotenv()
    validate_api_key()
    return OpenAI()


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    client = get_client()
    corpus = build_corpus_from_chunks(chunks, client)
    save_corpus_to_disk(corpus, INDEX_PATH, METADATA_PATH)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved chunk metadata to {METADATA_PATH}")
    print(f"Indexed {corpus.chunk_count} chunks with {corpus.index.d} dimensions")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
