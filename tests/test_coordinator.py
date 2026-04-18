import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent import ReceiptClassificationResult
from coordinator import (
    DirectoryCoordinator,
    Progress,
    load_progress,
    save_progress,
    SUPPORTED_EXTENSIONS,
)
from folder_agent import FolderClassificationResult


def test_coordinator_skips_folder_when_agent_says_skip(tmp_path):
    # Folder agent is only called on subdirectories, not the root.
    # Create a subdirectory that the agent will mark as skip.
    subdir = tmp_path / "project"
    subdir.mkdir()
    (subdir / "file.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    def fake_classify_folder(agent, path):
        return FolderClassificationResult(skip=True, reason="project folder")

    with patch("coordinator.classify_folder", side_effect=fake_classify_folder), \
         patch("coordinator.classify_file") as mock_file, \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)
        mock_file.assert_not_called()

    assert str(subdir) in progress.skipped_dirs


def test_coordinator_processes_files_when_not_skipped(tmp_path):
    (tmp_path / "receipt.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()
    receipt = ReceiptClassificationResult(
        is_receipt=True,
        reason="invoice",
        category="Travel",
        suggested_path="~/Documents/Receipts/Travel/receipt.pdf",
    )

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file", return_value=receipt), \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)

    csv_writer.append_receipt.assert_called_once()


def test_coordinator_skips_already_processed_files(tmp_path):
    f = tmp_path / "file.pdf"
    f.write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress(files={str(f)})

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file") as mock_file, \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)
        mock_file.assert_not_called()


def test_coordinator_recurses_into_subdirectories(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    folder_call_paths = []

    def fake_classify_folder(agent, path):
        folder_call_paths.append(path)
        return FolderClassificationResult(skip=False, reason="mixed")

    with patch("coordinator.classify_folder", side_effect=fake_classify_folder), \
         patch("coordinator.classify_file", return_value=None), \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)

    # Root is never passed to classify_folder — only subdirectories are classified.
    assert len(folder_call_paths) == 1
    assert subdir in folder_call_paths
    assert tmp_path not in folder_call_paths


def test_coordinator_skips_previously_skipped_dirs(tmp_path):
    progress = Progress(skipped_dirs={str(tmp_path)})
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()

    with patch("coordinator.classify_folder") as mock_folder:
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)
        mock_folder.assert_not_called()


def test_coordinator_ignores_hidden_files(tmp_path):
    (tmp_path / ".hidden.pdf").write_bytes(b"")
    (tmp_path / "visible.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    classified = []

    def fake_classify_file(agent, path):
        classified.append(path.name)
        return None

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file", side_effect=fake_classify_file), \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)

    assert "visible.pdf" in classified
    assert ".hidden.pdf" not in classified


def test_coordinator_ignores_unsupported_extensions(tmp_path):
    (tmp_path / "image.png").write_bytes(b"")
    (tmp_path / "receipt.pdf").write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    classified = []

    def fake_classify_file(agent, path):
        classified.append(path.name)
        return None

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file", side_effect=fake_classify_file), \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)

    assert "receipt.pdf" in classified
    assert "image.png" not in classified


def test_load_progress_returns_empty_when_no_file(tmp_path):
    progress = load_progress(str(tmp_path / ".progress.json"))
    assert len(progress.files) == 0
    assert len(progress.skipped_dirs) == 0


def test_load_progress_backward_compat_with_plain_list(tmp_path):
    progress_file = tmp_path / ".progress.json"
    progress_file.write_text(json.dumps(["/path/to/file.pdf"]))

    progress = load_progress(str(progress_file))

    assert "/path/to/file.pdf" in progress.files
    assert len(progress.skipped_dirs) == 0


def test_load_progress_new_format(tmp_path):
    progress_file = tmp_path / ".progress.json"
    progress_file.write_text(json.dumps({
        "files": ["/path/to/file.pdf"],
        "skipped_dirs": ["/path/to/B-29"],
    }))

    progress = load_progress(str(progress_file))

    assert "/path/to/file.pdf" in progress.files
    assert "/path/to/B-29" in progress.skipped_dirs


def test_coordinator_handles_permission_error(tmp_path):
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file") as mock_file, \
         patch("coordinator.save_progress"), \
         patch.object(Path, "iterdir", side_effect=PermissionError("access denied")):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)  # Should not raise
        mock_file.assert_not_called()


def test_coordinator_records_processed_file_in_progress(tmp_path):
    f = tmp_path / "invoice.pdf"
    f.write_bytes(b"")
    folder_agent = MagicMock()
    file_agent = MagicMock()
    csv_writer = MagicMock()
    progress = Progress()

    with patch("coordinator.classify_folder", return_value=FolderClassificationResult(skip=False, reason="mixed")), \
         patch("coordinator.classify_file", return_value=None), \
         patch("coordinator.save_progress"):
        coord = DirectoryCoordinator(folder_agent, file_agent, csv_writer, progress)
        coord.process(tmp_path)

    assert str(f) in progress.files


def test_save_progress_writes_both_sets(tmp_path):
    progress_file = tmp_path / ".progress.json"
    progress = Progress(
        files={"/path/to/file.pdf"},
        skipped_dirs={"/path/to/B-29"},
    )

    save_progress(progress, str(progress_file))

    data = json.loads(progress_file.read_text())
    assert "/path/to/file.pdf" in data["files"]
    assert "/path/to/B-29" in data["skipped_dirs"]
