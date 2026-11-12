from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf


def extract_text_from_pdf(pdf_path: str | Path, ocr: bool = False) -> str:
    """Extract the text of all pages of a PDF with PyMuPDF.

    ocr=True uses PyMuPDF's OCR (needs Tesseract installed); scanned PDFs have
    no text layer, so without OCR their text comes out empty.
    """
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            if ocr:
                textpage = page.get_textpage_ocr(dpi=200, full=True)
                pages.append(page.get_text(textpage=textpage))
            else:
                pages.append(page.get_text())
    return "\n".join(pages)


def extract_pdf_dataset(
    pdf_dir: str | Path,
    samples: list[dict[str, Any]],
    suffix: str = "",
    ocr: bool = False,
) -> list[dict[str, Any]]:
    """For each JSONL record, read PDF_DIR/<id><suffix>.pdf and replace 'text' with the PDF text.

    Stops at the first missing PDF, so the result is a prefix of the dataset
    (evaluation.py compares against the matching prefix of the gold file).
    """
    pdf_samples = []
    for sample in samples:
        pdf_path = Path(pdf_dir) / f"{sample['id']}{suffix}.pdf"
        if not pdf_path.exists():
            break
        pdf_samples.append({**sample, "text": extract_text_from_pdf(pdf_path, ocr=ocr)})
    return pdf_samples
