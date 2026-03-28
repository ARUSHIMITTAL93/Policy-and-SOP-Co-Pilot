from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
INDEX_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "chunk_metadata.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a list of chunk records")

    return chunks


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


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    embeddings: list[list[float]] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
        )

        ordered_data = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in ordered_data)

    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.size == 0:
        raise ValueError("No embeddings were created from the chunk data")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_metadata(chunks: list[dict]) -> None:
    with METADATA_PATH.open("wb") as metadata_file:
        pickle.dump(chunks, metadata_file)


def main() -> None:
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    client = get_client()
    embeddings = embed_texts(client, texts)
    index = build_faiss_index(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    save_metadata(chunks)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved chunk metadata to {METADATA_PATH}")
    print(f"Indexed {len(chunks)} chunks with {embeddings.shape[1]} dimensions")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
