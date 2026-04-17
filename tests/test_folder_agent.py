from pathlib import Path
from unittest.mock import MagicMock

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


def test_classify_folder_returns_skip_true():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": '{"skip": true, "reason": "Numbered B-29 technical drawings."}',
        "intermediate_steps": [],
    }
    result = classify_folder(mock_executor, Path("/fake/B-29"))
    assert result.skip is True
    assert result.reason == "Numbered B-29 technical drawings."


def test_classify_folder_returns_skip_false():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": '{"skip": false, "reason": "Mixed content, generic folder."}',
        "intermediate_steps": [],
    }
    result = classify_folder(mock_executor, Path("/fake/Downloads"))
    assert result.skip is False
    assert "Mixed content" in result.reason


def test_classify_folder_defaults_to_process_on_malformed_json():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": "I cannot classify this folder.",
        "intermediate_steps": [],
    }
    result = classify_folder(mock_executor, Path("/fake/unknown"))
    assert result.skip is False


def test_classify_folder_defaults_to_process_on_agent_exception():
    mock_executor = MagicMock()
    mock_executor.invoke.side_effect = Exception("Ollama connection refused")
    result = classify_folder(mock_executor, Path("/fake/folder"))
    assert result.skip is False
