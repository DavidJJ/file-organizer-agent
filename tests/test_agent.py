from pathlib import Path
from unittest.mock import MagicMock
from agent import _parse_json_output, classify_file, ReceiptClassificationResult


def test_parse_json_output_valid_receipt():
    output = '{"is_receipt": true, "category": "Hobby / Radio Control", "reason": "HobbyKing invoice.", "suggested_path": "~/Documents/Receipts/Hobby_RC/order.pdf"}'
    result = _parse_json_output(output)

    assert result is not None
    assert result["is_receipt"] is True
    assert result["category"] == "Hobby / Radio Control"
    assert result["reason"] == "HobbyKing invoice."


def test_parse_json_output_not_receipt():
    output = '{"is_receipt": false, "reason": "This is a user manual."}'
    result = _parse_json_output(output)

    assert result is not None
    assert result["is_receipt"] is False
    assert result["reason"] == "This is a user manual."


def test_parse_json_output_strips_markdown_fences():
    output = '```json\n{"is_receipt": false, "reason": "Not a receipt."}\n```'
    result = _parse_json_output(output)

    assert result is not None
    assert result["is_receipt"] is False


def test_parse_json_output_returns_none_on_invalid():
    result = _parse_json_output("Sorry, I cannot determine this from the file.")
    assert result is None


def test_classify_file_returns_receipt_result():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": '{"is_receipt": true, "category": "Travel", "reason": "Flight booking.", "suggested_path": "~/Documents/Receipts/Travel/ticket.pdf"}'
    }

    result = classify_file(mock_executor, Path("/fake/ticket.pdf"))

    assert result is not None
    assert result.is_receipt is True
    assert result.category == "Travel"
    assert result.reason == "Flight booking."
    assert result.suggested_path == "~/Documents/Receipts/Travel/ticket.pdf"


def test_classify_file_returns_non_receipt_result():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {
        "output": '{"is_receipt": false, "reason": "This is a user manual."}'
    }

    result = classify_file(mock_executor, Path("/fake/manual.pdf"))

    assert result is not None
    assert result.is_receipt is False
    assert result.reason == "This is a user manual."


def test_classify_file_returns_none_on_malformed_json():
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {"output": "I cannot determine this."}

    result = classify_file(mock_executor, Path("/fake/unknown.pdf"))

    assert result is None


def test_classify_file_returns_none_on_agent_exception():
    mock_executor = MagicMock()
    mock_executor.invoke.side_effect = Exception("Ollama connection refused")

    result = classify_file(mock_executor, Path("/fake/file.pdf"))

    assert result is None
