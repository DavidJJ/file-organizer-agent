import csv
from output import CSVWriter

FIELDNAMES = ["Receipt Category", "Original File Location", "Suggested File Location", "Reason"]


def test_csv_writer_creates_file_with_header(tmp_path):
    csv_path = str(tmp_path / "receipts.csv")
    CSVWriter(csv_path)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDNAMES


def test_csv_writer_appends_receipt(tmp_path):
    csv_path = str(tmp_path / "receipts.csv")
    writer = CSVWriter(csv_path)

    writer.append_receipt(
        category="Hobby / Radio Control",
        original_path="/Users/david/Downloads/hobbyking.pdf",
        suggested_path="~/Documents/Receipts/Hobby_RC/hobbyking.pdf",
        reason="Invoice from HobbyKing with itemized parts.",
    )

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["Receipt Category"] == "Hobby / Radio Control"
    assert rows[0]["Original File Location"] == "/Users/david/Downloads/hobbyking.pdf"
    assert rows[0]["Suggested File Location"] == "~/Documents/Receipts/Hobby_RC/hobbyking.pdf"
    assert rows[0]["Reason"] == "Invoice from HobbyKing with itemized parts."


def test_csv_writer_appends_multiple_rows(tmp_path):
    csv_path = str(tmp_path / "receipts.csv")
    writer = CSVWriter(csv_path)

    writer.append_receipt("Mortgage", "/path/a.pdf", "~/Documents/Receipts/Mortgage/a.pdf", "Monthly payment.")
    writer.append_receipt("Travel", "/path/b.pdf", "~/Documents/Receipts/Travel/b.pdf", "Flight booking.")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["Receipt Category"] == "Mortgage"
    assert rows[1]["Receipt Category"] == "Travel"


def test_csv_writer_survives_reinit_on_existing_file(tmp_path):
    """Each CSVWriter call creates a fresh file (timestamps make paths unique in practice)."""
    csv_path = str(tmp_path / "receipts.csv")
    writer = CSVWriter(csv_path)
    writer.append_receipt("Mortgage", "/a.pdf", "~/Documents/Receipts/Mortgage/a.pdf", "Payment.")

    CSVWriter(csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 0
