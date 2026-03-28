from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "chunks.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


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


def load_processed_documents() -> list[dict]:
    documents = []

    for json_path in sorted(PROCESSED_DIR.glob("*.json")):
        if json_path == OUTPUT_PATH:
            continue

        document = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "pages" in document:
            documents.append(document)

    return documents


def create_chunks() -> list[dict]:
    all_chunks = []

    for document in load_processed_documents():
        source_file = document.get("filename", "unknown.pdf")
        document_title = document.get("document_title") or Path(source_file).stem

        for page in document.get("pages", []):
            page_number = int(page.get("page_number", 0))
            page_text = page.get("text", "")
            page_chunks = split_text_into_chunks(page_text)

            for chunk_index_on_page, chunk_text in enumerate(page_chunks, start=1):
                all_chunks.append(
                    {
                        "chunk_id": build_chunk_id(
                            source_file, page_number, chunk_index_on_page
                        ),
                        "source_file": source_file,
                        "document_title": document_title,
                        "page_number": page_number,
                        "chunk_index_on_page": chunk_index_on_page,
                        "text": chunk_text,
                    }
                )

    return all_chunks


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks = create_chunks()
    OUTPUT_PATH.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {len(chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
