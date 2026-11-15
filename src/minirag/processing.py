from itertools import batched
from pathlib import Path
from minirag.documents import load_pdf, load_text


def chunk_text(text: str, chunk_size: int = 10) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    words = text.split()
    chunks = batched(words, chunk_size)
    return [" ".join(chunk) for chunk in chunks]


def chunk_text_old_ver(text: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        part = words[i : i + chunk_size]
        chunk = " ".join(part)
        chunks.append(chunk)
    return chunks


def process_document(path: str | Path, chunk_size: int = 10) -> list[str]:
    file_path = Path(path)
    file_extension = file_path.suffix.lower()
    if file_extension not in (".txt", ".pdf"):
        raise ValueError("Unsuported file format")
    if file_extension == ".txt":
        text = load_text(file_path)
    if file_extension == ".pdf":
        text = load_pdf(file_path)
    return chunk_text(text, chunk_size)
