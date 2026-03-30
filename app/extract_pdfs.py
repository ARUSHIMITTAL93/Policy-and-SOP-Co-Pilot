from __future__ import annotations

from pathlib import Path

from document_pipeline import extract_pdf_from_path, save_extracted_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def extract_pdf(pdf_path: Path) -> dict:
    return extract_pdf_from_path(pdf_path)


def save_extracted_pdf(pdf_path: Path) -> Path:
    extracted_document = extract_pdf(pdf_path)
    output_path = PROCESSED_DIR / f"{pdf_path.stem}.json"
    return save_extracted_document(extracted_document, output_path)


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
