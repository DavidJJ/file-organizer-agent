from unittest.mock import patch, MagicMock


def test_read_pdf_returns_text():
    from tools.read_pdf import read_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Order Total: $29.99\nItem: Widget"
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("tools.read_pdf.pdfplumber.open", return_value=mock_pdf):
        result = read_pdf.invoke("/fake/path/receipt.pdf")

    assert "Order Total" in result
    assert "Widget" in result


def test_read_pdf_handles_error():
    from tools.read_pdf import read_pdf

    with patch("tools.read_pdf.pdfplumber.open", side_effect=Exception("corrupt file")):
        result = read_pdf.invoke("/fake/path/broken.pdf")

    assert "Error reading PDF" in result


def test_read_pdf_truncates_long_text():
    from tools.read_pdf import read_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "x" * 5000
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("tools.read_pdf.pdfplumber.open", return_value=mock_pdf):
        result = read_pdf.invoke("/fake/path/long.pdf")

    assert len(result) <= 3000


def test_read_pdf_handles_no_text():
    from tools.read_pdf import read_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = None
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("tools.read_pdf.pdfplumber.open", return_value=mock_pdf):
        result = read_pdf.invoke("/fake/path/empty.pdf")

    assert "No text" in result


def test_read_docx_returns_text():
    from tools.read_docx import read_docx

    mock_para1 = MagicMock()
    mock_para1.text = "Invoice #12345"
    mock_para2 = MagicMock()
    mock_para2.text = "Amount Due: $100.00"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para1, mock_para2]

    with patch("tools.read_docx.Document", return_value=mock_doc):
        result = read_docx.invoke("/fake/path/invoice.docx")

    assert "Invoice #12345" in result
    assert "Amount Due: $100.00" in result


def test_read_docx_handles_error():
    from tools.read_docx import read_docx

    with patch("tools.read_docx.Document", side_effect=Exception("bad file")):
        result = read_docx.invoke("/fake/path/broken.docx")

    assert "Error reading DOCX" in result


def test_read_text_returns_content(tmp_path):
    from tools.read_text import read_text

    test_file = tmp_path / "receipt.txt"
    test_file.write_text("Purchase confirmation\nTotal: $50.00")

    result = read_text.invoke(str(test_file))

    assert "Purchase confirmation" in result
    assert "Total: $50.00" in result


def test_read_text_truncates_long_content(tmp_path):
    from tools.read_text import read_text

    test_file = tmp_path / "long.txt"
    test_file.write_text("x" * 5000)

    result = read_text.invoke(str(test_file))

    assert len(result) <= 3000


def test_read_text_handles_missing_file():
    from tools.read_text import read_text

    result = read_text.invoke("/nonexistent/path/file.txt")

    assert "Error reading file" in result
