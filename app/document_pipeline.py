from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import fitz
import numpy as np
from openai import OpenAI
from tqdm import tqdm


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 32


@dataclass
class DocumentCorpus:
    index: faiss.Index
    metadata: list[dict]
    document_names: list[str]
    chunk_count: int
    document_count: int


def extract_pdf_document(filename: str, pdf_bytes: bytes) -> dict:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        metadata = document.metadata or {}
        document_title = (metadata.get("title") or "").strip() or Path(filename).stem

        pages = []
        for page_index, page in enumerate(document, start=1):
            pages.append(
                {
                    "page_number": page_index,
                    "text": page.get_text("text").strip(),
                }
            )

        return {
            "filename": filename,
            "document_title": document_title,
            "total_pages": document.page_count,
            "pages": pages,
        }
    finally:
        document.close()


def extract_pdf_from_path(pdf_path: Path) -> dict:
    return extract_pdf_document(pdf_path.name, pdf_path.read_bytes())


def save_extracted_document(document: dict, output_path: Path) -> Path:
    output_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def load_processed_documents(
    processed_dir: Path, skip_paths: set[Path] | None = None
) -> list[dict]:
    skip_paths = skip_paths or set()
    documents = []

    for json_path in sorted(processed_dir.glob("*.json")):
        if json_path in skip_paths:
            continue

        document = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "pages" in document:
            documents.append(document)

    return documents


def align_chunk_start(text: str, start: int, lower_bound: int) -> int:
    start = max(start, lower_bound)
    if start <= 0 or start >= len(text):
        return max(0, min(start, len(text)))

    if not text[start].isspace() and not text[start - 1].isspace():
        newline_break = text.rfind("\n", lower_bound - 1, start)
        space_break = text.rfind(" ", lower_bound - 1, start)
        chosen_break = max(newline_break, space_break)

        if chosen_break != -1:
            start = chosen_break + 1
        else:
            next_newline = text.find("\n", start)
            next_space = text.find(" ", start)
            next_breaks = [index for index in (next_newline, next_space) if index != -1]

            if next_breaks:
                start = min(next_breaks) + 1

    while start < len(text) and text[start].isspace():
        start += 1

    return start


def split_text_into_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    minimum_breakpoint = int(chunk_size * 0.6)

    while start < len(cleaned_text):
        ideal_end = min(start + chunk_size, len(cleaned_text))
        end = ideal_end

        if ideal_end < len(cleaned_text):
            search_start = min(start + minimum_breakpoint, ideal_end)
            newline_break = cleaned_text.rfind("\n", search_start, ideal_end)
            space_break = cleaned_text.rfind(" ", search_start, ideal_end)
            chosen_break = max(newline_break, space_break)

            if chosen_break > start:
                end = chosen_break

        chunk_text = cleaned_text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(cleaned_text):
            break

        next_start = max(end - chunk_overlap, start + 1)
        start = align_chunk_start(cleaned_text, next_start, start + 1)

    return chunks


def build_chunk_id(source_file: str, page_number: int, chunk_index_on_page: int) -> str:
    source_stem = Path(source_file).stem
    return f"{source_stem}_p{page_number}_c{chunk_index_on_page}"


def create_chunks_from_document(document: dict) -> list[dict]:
    chunks = []
    source_file = document.get("filename", "unknown.pdf")
    document_title = document.get("document_title") or Path(source_file).stem

    for page in document.get("pages", []):
        page_number = int(page.get("page_number", 0))
        page_text = page.get("text", "")
        page_chunks = split_text_into_chunks(page_text)

        for chunk_index_on_page, chunk_text in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": build_chunk_id(source_file, page_number, chunk_index_on_page),
                    "source_file": source_file,
                    "document_title": document_title,
                    "page_number": page_number,
                    "chunk_index_on_page": chunk_index_on_page,
                    "text": chunk_text,
                }
            )

    return chunks


def create_chunks_from_documents(documents: list[dict]) -> list[dict]:
    chunks = []

    for document in documents:
        chunks.extend(create_chunks_from_document(document))

    return chunks


def save_chunks(chunks: list[dict], output_path: Path) -> Path:
    output_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def load_chunks(chunks_path: Path) -> list[dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a list of chunk records")

    return chunks


def embed_texts(
    client: OpenAI,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    embeddings: list[list[float]] = []
    ranges = range(0, len(texts), batch_size)
    iterator = tqdm(ranges, desc="Embedding chunks") if show_progress else ranges

    for start in iterator:
        batch = texts[start : start + batch_size]
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


def build_corpus_from_chunks(
    chunks: list[dict], client: OpenAI, show_progress: bool = True
) -> DocumentCorpus:
    if not chunks:
        raise ValueError("No chunks were provided to build the corpus")

    embeddings = embed_texts(
        client=client,
        texts=[chunk["text"] for chunk in chunks],
        show_progress=show_progress,
    )
    index = build_faiss_index(embeddings)
    document_names = sorted({chunk["source_file"] for chunk in chunks})

    return DocumentCorpus(
        index=index,
        metadata=chunks,
        document_names=document_names,
        chunk_count=len(chunks),
        document_count=len(document_names),
    )


def save_corpus_to_disk(
    corpus: DocumentCorpus, index_path: Path, metadata_path: Path
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(corpus.index, str(index_path))

    with metadata_path.open("wb") as metadata_file:
        pickle.dump(corpus.metadata, metadata_file)


def load_corpus_from_disk(index_path: Path, metadata_path: Path) -> DocumentCorpus:
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. Run app/build_index.py first."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found: {metadata_path}. Run app/build_index.py first."
        )

    index = faiss.read_index(str(index_path))

    with metadata_path.open("rb") as metadata_file:
        metadata = pickle.load(metadata_file)

    if not isinstance(metadata, list):
        raise ValueError("chunk_metadata.pkl must contain a list of chunk records")

    document_names = sorted({chunk["source_file"] for chunk in metadata})

    return DocumentCorpus(
        index=index,
        metadata=metadata,
        document_names=document_names,
        chunk_count=len(metadata),
        document_count=len(document_names),
    )
