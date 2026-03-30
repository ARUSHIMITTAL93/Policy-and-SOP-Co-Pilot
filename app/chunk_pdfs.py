from __future__ import annotations

from pathlib import Path

from document_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    align_chunk_start,
    build_chunk_id,
    create_chunks_from_documents,
    load_processed_documents,
    save_chunks,
    split_text_into_chunks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "chunks.json"


def create_chunks() -> list[dict]:
    documents = load_processed_documents(PROCESSED_DIR, skip_paths={OUTPUT_PATH})
    return create_chunks_from_documents(documents)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks = create_chunks()
    save_chunks(chunks, OUTPUT_PATH)
    print(f"Saved {len(chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
