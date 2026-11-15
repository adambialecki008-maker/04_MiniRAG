from pathlib import Path
from pypdf import PdfReader


def load_text(path: str | Path) -> str:
    file_path = Path(path)
    validate_file_path(file_path)
    return file_path.read_text(encoding="utf-8")


def load_pdf(path: str | Path) -> str:
    file_path = Path(path)
    validate_file_path(file_path)
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def validate_file_path(file_path: Path) -> None:
    if file_path.is_dir():
        raise ValueError("Expected a file, got a directory")
    if not file_path.exists():
        raise FileNotFoundError(file_path)
