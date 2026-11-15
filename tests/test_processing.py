import pytest
from minirag.processing import chunk_text, process_document
from pathlib import Path


@pytest.mark.parametrize(
    "text,chunk_size,expected",
    [
        ("A B C D E F", 2, ["A B", "C D", "E F"]),
        ("A B C D E", 2, ["A B", "C D", "E"]),
        ("A B C", 10, ["A B C"]),
        ("", 3, []),
    ],
)
def test_chunk_text_success(text, chunk_size, expected):
    assert chunk_text(text, chunk_size) == expected


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_text_raises_on_invalid_size(chunk_size):
    with pytest.raises(ValueError):
        chunk_text("", chunk_size)


def test_process_document_success_with_txt_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "PLC controls motor via Siemens Variable Frequency Drive.", encoding="utf-8"
    )
    chunks = process_document(file_path, 3)
    assert len(chunks) == 3
    assert isinstance(chunks, list)
    assert chunks == [
        "PLC controls motor",
        "via Siemens Variable",
        "Frequency Drive.",
    ]


def test_process_document_success_with_pdf_file():
    pdf_path = Path("documents/sample.pdf")
    chunks = process_document(pdf_path, 50)
    assert len(chunks) == 9
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk.split()) <= 50


def test_process_document_raises_on_wrong_file_type(tmp_path):
    file_path = tmp_path / "sample.docx"
    with pytest.raises(ValueError):
        process_document(file_path)
