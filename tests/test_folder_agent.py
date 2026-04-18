from pathlib import Path
from unittest.mock import MagicMock, patch

from folder_agent import FolderClassificationResult, classify_folder, _parse_json_output


def test_parse_json_output_skip():
    output = '{"skip": true, "reason": "All files are B-29 drawings."}'
    result = _parse_json_output(output)
    assert result is not None
    assert result["skip"] is True
    assert result["reason"] == "All files are B-29 drawings."


def test_parse_json_output_process():
    output = '{"skip": false, "reason": "Mixed files, generic folder name."}'
    result = _parse_json_output(output)
    assert result is not None
    assert result["skip"] is False


def test_parse_json_output_strips_markdown_fences():
    output = '```json\n{"skip": true, "reason": "Project folder."}\n```'
    result = _parse_json_output(output)
    assert result is not None
    assert result["skip"] is True


def test_parse_json_output_returns_none_on_invalid():
    result = _parse_json_output("I cannot determine this.")
    assert result is None


def _make_mock_llm(json_output: str) -> MagicMock:
    """Return a mock ChatOllama whose .invoke() returns an object with .content."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json_output
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def test_classify_folder_returns_skip_true(tmp_path):
    mock_llm = _make_mock_llm('{"skip": true, "reason": "Numbered B-29 technical drawings."}')
    with patch("folder_agent.list_directory") as mock_listing:
        mock_listing.invoke.return_value = "Folder name: B-29\nContents (2 items):\n  [file]  B-29-1828-WingSpars.pdf\n  [file]  B-29-1829-Fuselage.pdf"
        result = classify_folder(mock_llm, Path("/fake/B-29"))
    assert result.skip is True
    assert result.reason == "Numbered B-29 technical drawings."


def test_classify_folder_returns_skip_false(tmp_path):
    mock_llm = _make_mock_llm('{"skip": false, "reason": "Mixed content, generic folder."}')
    with patch("folder_agent.list_directory") as mock_listing:
        mock_listing.invoke.return_value = "Folder name: Downloads\nContents (3 items):\n  [file]  invoice.pdf\n  [file]  readme.txt\n  [dir]   project"
        result = classify_folder(mock_llm, Path("/fake/Downloads"))
    assert result.skip is False
    assert "Mixed content" in result.reason


def test_classify_folder_defaults_to_process_on_malformed_json():
    mock_llm = _make_mock_llm("I cannot classify this folder.")
    with patch("folder_agent.list_directory") as mock_listing:
        mock_listing.invoke.return_value = "Folder name: unknown\nContents (1 items):\n  [file]  file.pdf"
        result = classify_folder(mock_llm, Path("/fake/unknown"))
    assert result.skip is False


def test_classify_folder_defaults_to_process_on_llm_exception():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Ollama connection refused")
    with patch("folder_agent.list_directory") as mock_listing:
        mock_listing.invoke.return_value = "Folder name: folder\nContents (1 items):\n  [file]  file.pdf"
        result = classify_folder(mock_llm, Path("/fake/folder"))
    assert result.skip is False


def test_classify_folder_calls_list_directory_with_path():
    """Verify the listing is always fetched in Python, never delegated to the LLM."""
    mock_llm = _make_mock_llm('{"skip": false, "reason": "Mixed."}')
    with patch("folder_agent.list_directory") as mock_listing:
        mock_listing.invoke.return_value = "Folder name: misc\nContents (0 items):\n  (empty)"
        classify_folder(mock_llm, Path("/Users/david/Downloads/misc"))
        mock_listing.invoke.assert_called_once_with("/Users/david/Downloads/misc")


def test_classify_folder_listing_included_in_llm_prompt():
    """Verify the directory listing is passed to the LLM prompt."""
    listing_text = "Folder name: B-29\nContents (2 items):\n  [file]  B-29-1828-WingSpars.pdf"
    mock_llm = _make_mock_llm('{"skip": true, "reason": "Project folder."}')
    with patch("folder_agent.list_directory") as mock_listing:
        mock_listing.invoke.return_value = listing_text
        classify_folder(mock_llm, Path("/fake/B-29"))
        call_args = mock_llm.invoke.call_args[0][0]
        assert listing_text in call_args
