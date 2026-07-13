import io

from pypdf import PdfReader

SUPPORTED_TYPES = {".pdf", ".txt", ".md"}


def extract_text(filename: str, content: bytes) -> str:
    suffix = _suffix(filename)
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix}")


def _suffix(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx != -1 else ""
