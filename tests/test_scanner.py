from pathlib import Path
import pytest
from scanner import scan_files


def test_scan_finds_pdf_files(tmp_path):
    (tmp_path / "invoice.pdf").write_bytes(b"fake pdf")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "image.png").write_bytes(b"fake png")

    results = list(scan_files(str(tmp_path)))
    names = [p.name for p in results]

    assert "invoice.pdf" in names
    assert "notes.txt" in names
    assert "image.png" not in names


def test_scan_skips_hidden_files(tmp_path):
    (tmp_path / ".hidden.pdf").write_bytes(b"fake pdf")
    (tmp_path / "visible.pdf").write_bytes(b"fake pdf")

    results = list(scan_files(str(tmp_path)))
    names = [p.name for p in results]

    assert ".hidden.pdf" not in names
    assert "visible.pdf" in names


def test_scan_skips_hidden_directories(tmp_path):
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir()
    (hidden_dir / "receipt.pdf").write_bytes(b"fake pdf")
    (tmp_path / "visible.pdf").write_bytes(b"fake pdf")

    results = list(scan_files(str(tmp_path)))
    paths = [str(p) for p in results]

    assert not any(".hidden_dir" in p for p in paths)
    assert any("visible.pdf" in p for p in paths)


def test_scan_walks_subdirectories(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.pdf").write_bytes(b"fake pdf")

    results = list(scan_files(str(tmp_path)))
    names = [p.name for p in results]

    assert "nested.pdf" in names


def test_scan_finds_docx_and_txt(tmp_path):
    (tmp_path / "doc.docx").write_bytes(b"fake docx")
    (tmp_path / "old.doc").write_bytes(b"fake doc")
    (tmp_path / "note.txt").write_text("text")

    results = list(scan_files(str(tmp_path)))
    names = [p.name for p in results]

    assert "doc.docx" in names
    assert "old.doc" in names
    assert "note.txt" in names


def test_scan_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "file.pdf").write_bytes(b"fake pdf")

    results = list(scan_files("~"))
    names = [p.name for p in results]

    assert "file.pdf" in names
