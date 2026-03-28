from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "chunk_metadata.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve the most relevant document chunks for a question."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Question to search for across the indexed chunks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matching chunks to return.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON instead of formatted text.",
    )
    return parser.parse_args()


def validate_api_key() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("Set OPENAI_API_KEY in .env before running retrieval.")


def get_client() -> OpenAI:
    load_dotenv()
    validate_api_key()
    return OpenAI()


def load_index() -> faiss.Index:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}. Run app/build_index.py first."
        )

    return faiss.read_index(str(INDEX_PATH))


def load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found: {METADATA_PATH}. Run app/build_index.py first."
        )

    with METADATA_PATH.open("rb") as metadata_file:
        metadata = pickle.load(metadata_file)

    if not isinstance(metadata, list):
        raise ValueError("chunk_metadata.pkl must contain a list of chunk records")

    return metadata


def embed_query(client: OpenAI, question: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
        encoding_format="float",
    )
    query_vector = np.asarray(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)
    return query_vector


def retrieve_chunks(question: str, top_k: int = 5) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    index = load_index()
    metadata = load_metadata()

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match chunk metadata count. Rebuild the index."
        )

    client = get_client()
    query_vector = embed_query(client, question)

    if query_vector.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension does not match the stored FAISS index. "
            "Rebuild the index with the same embedding model."
        )

    search_k = min(top_k, len(metadata))
    scores, indices = index.search(query_vector, search_k)

    results = []
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index == -1:
            continue

        chunk = metadata[chunk_index].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def format_results(question: str, results: list[dict]) -> str:
    if not results:
        return f'No results found for: "{question}"'

    lines = [f'Question: "{question}"', ""]

    for rank, result in enumerate(results, start=1):
        lines.extend(
            [
                f"Result {rank}",
                f"Score: {result['score']:.4f}",
                f"Source file: {result['source_file']}",
                f"Document title: {result['document_title']}",
                f"Page number: {result['page_number']}",
                f"Chunk id: {result['chunk_id']}",
                "Text:",
                result["text"],
                "",
            ]
        )

    return "\n".join(lines).rstrip()


def main() -> None:
    args = parse_args()
    question = " ".join(args.question).strip()

    try:
        results = retrieve_chunks(question=question, top_k=args.top_k)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(format_results(question, results))


if __name__ == "__main__":
    main()
