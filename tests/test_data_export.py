import unittest
from datetime import datetime, timezone
from io import BytesIO
import json
from types import SimpleNamespace

from openpyxl import load_workbook

from data_export import (
    MAX_CELL_TEXT_LENGTH,
    build_export_workbook,
    fetch_all_table_rows,
)


class FakeQuery:
    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.start = 0
        self.end = -1

    def select(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def range(self, start, end):
        self.start = start
        self.end = end
        return self

    def execute(self):
        self.client.ranges.append((self.table_name, self.start, self.end))
        return SimpleNamespace(data=self.client.rows[self.start:self.end + 1])


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.ranges = []

    def table(self, table_name):
        return FakeQuery(self, table_name)


class DataExportTests(unittest.TestCase):
    def test_fetches_every_page_in_stable_ranges(self):
        rows = [{"id": index, "name": f"Item {index}"} for index in range(1201)]
        client = FakeClient(rows)

        exported = fetch_all_table_rows(client, "inventory_items", page_size=500)

        self.assertEqual(exported, rows)
        self.assertEqual(
            client.ranges,
            [
                ("inventory_items", 0, 499),
                ("inventory_items", 500, 999),
                ("inventory_items", 1000, 1499),
            ],
        )

    def test_builds_a_readable_xlsx_with_timestamp_nested_data_and_empty_headers(self):
        exported_at = datetime(2026, 11, 18, 12, 30, tzinfo=timezone.utc)
        workbook = build_export_workbook(
            {
                "Inventory": [
                    {
                        "id": 7,
                        "name": "=1+1",
                        "custom_values": {"supplier": "Local", "lot": 2},
                    }
                ],
                "Events": [],
            },
            exported_at=exported_at,
        )
        output = BytesIO()
        workbook.save(output)
        output.seek(0)
        workbook = load_workbook(output, data_only=False)

        self.assertEqual(workbook.sheetnames, ["Export Info", "Inventory", "Events"])
        self.assertEqual(workbook["Export Info"]["B2"].value, "2026-11-18T12:30:00+00:00")
        self.assertEqual(workbook["Inventory"]["A2"].value, 7)
        self.assertEqual(workbook["Inventory"]["B2"].value, "=1+1")
        self.assertEqual(workbook["Inventory"]["B2"].data_type, "s")
        self.assertEqual(
            json.loads(workbook["Inventory"]["C2"].value),
            {"supplier": "Local", "lot": 2},
        )
        self.assertEqual(workbook["Events"]["A1"].value, "id")

    def test_long_values_are_split_across_cells_without_truncation(self):
        long_note = "x" * (MAX_CELL_TEXT_LENGTH + 23)
        workbook = build_export_workbook({"Long Notes": [{"note": long_note}]})
        sheet = workbook["Long Notes"]

        self.assertEqual(sheet["A1"].value, "note (part 1)")
        self.assertEqual(sheet["B1"].value, "note (part 2)")
        self.assertEqual(sheet["A2"].value + sheet["B2"].value, long_note)


if __name__ == "__main__":
    unittest.main()
