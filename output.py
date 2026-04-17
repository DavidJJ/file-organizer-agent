import csv

FIELDNAMES = ["Receipt Category", "Original File Location", "Suggested File Location", "Reason"]


class CSVWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    def append_receipt(self, category: str, original_path: str, suggested_path: str, reason: str):
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({
                "Receipt Category": category,
                "Original File Location": original_path,
                "Suggested File Location": suggested_path,
                "Reason": reason,
            })
