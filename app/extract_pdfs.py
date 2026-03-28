from __future__ import annotations

import json
from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def extract_pdf(pdf_path: Path) -> dict:
    document = fitz.open(pdf_path)
    metadata = document.metadata or {}
    document_title = (metadata.get("title") or "").strip() or pdf_path.stem

    pages = []
    for page_index, page in enumerate(document, start=1):
        pages.append(
            {
                "page_number": page_index,
                "text": page.get_text("text").strip(),
            }
        )

    extracted_document = {
        "filename": pdf_path.name,
        "document_title": document_title,
        "total_pages": document.page_count,
        "pages": pages,
    }

    document.close()
    return extracted_document


def save_extracted_pdf(pdf_path: Path) -> Path:
    extracted_document = extract_pdf(pdf_path)
    output_path = PROCESSED_DIR / f"{pdf_path.stem}.json"
    output_path.write_text(
        json.dumps(extracted_document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {RAW_PDF_DIR}")
        return

    for pdf_path in pdf_files:
        output_path = save_extracted_pdf(pdf_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
