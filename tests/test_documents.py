from minirag.documents import load_text, load_pdf
import pytest
from pathlib import Path


def test_load_text_success(tmp_path):
    file_path = tmp_path / "test.txt"
    expected_text = "PLC controls motor"
    file_path.write_text(expected_text, encoding="utf-8")
    text = load_text(file_path)
    assert text == expected_text


def test_load_text_returns_empty_if_no_text_in_file(tmp_path):
    file_path = tmp_path / "test.txt"
    expected_text = ""
    file_path.write_text("", encoding="utf-8")
    text = load_text(file_path)
    assert text == expected_text


def test_load_text_raises_on_directory_path(tmp_path):
    folder = tmp_path / "docments"
    folder.mkdir()
    with pytest.raises(ValueError):
        load_text(folder)


def test_load_text_raises_on_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_text("test")


def test_load_pdf_succes():
    pdf_path = Path("documents/sample.pdf")
    expected_text = "Vivamus sollicitudin, metus ut interdum eleifend,"
    text = load_pdf(pdf_path)
    assert expected_text in text
    assert isinstance(text, str)


def test_load_pdf_raises_on_directory_path(tmp_path):
    folder = tmp_path / "docments"
    folder.mkdir()
    with pytest.raises(ValueError):
        load_pdf(folder)


def test_load_pdf_raises_on_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_pdf("test")
