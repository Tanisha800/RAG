import pymupdf as fitz

def extract_text_from_pdf(file_path: str) -> list:
    pages = []
    doc = fitz.open(file_path)

    for i, page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": i + 1,
                "text": text.strip()
            })

    return pages