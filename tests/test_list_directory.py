import pytest
from tools.list_directory import list_directory


def test_list_directory_returns_folder_name_and_contents(tmp_path):
    (tmp_path / "file1.pdf").write_bytes(b"")
    (tmp_path / "file2.txt").write_text("hello")
    subdir = tmp_path / "subproject"
    subdir.mkdir()

    result = list_directory.invoke(str(tmp_path))

    assert tmp_path.name in result
    assert "[file]  file1.pdf" in result
    assert "[file]  file2.txt" in result
    assert "[dir]  subproject" in result


def test_list_directory_excludes_hidden_entries(tmp_path):
    (tmp_path / ".hidden").write_text("")
    (tmp_path / "visible.pdf").write_bytes(b"")

    result = list_directory.invoke(str(tmp_path))

    assert ".hidden" not in result
    assert "visible.pdf" in result


def test_list_directory_handles_empty_dir(tmp_path):
    result = list_directory.invoke(str(tmp_path))
    assert "(empty)" in result


def test_list_directory_handles_nonexistent_path(tmp_path):
    result = list_directory.invoke(str(tmp_path / "nonexistent"))
    assert "Error" in result


def test_list_directory_shows_item_count(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "b.pdf").write_bytes(b"")

    result = list_directory.invoke(str(tmp_path))
    assert "2 items" in result


def test_list_directory_handles_non_directory_path(tmp_path):
    f = tmp_path / "file.pdf"
    f.write_bytes(b"")

    result = list_directory.invoke(str(f))
    assert "Error" in result
