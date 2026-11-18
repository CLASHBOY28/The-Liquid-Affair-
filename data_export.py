"""Fresh, paginated Excel exports for The Liquid Affair's application data."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# Every persisted application table is exported on each click.
EXPORT_TABLES = (
    ("Inventory", "inventory_items", (
        "id", "name", "category_id", "quantity", "reorder_threshold", "unit",
        "expiry_date", "custom_values", "updated_at",
    )),
    ("Categories", "categories", ("id", "name", "created_at")),
    ("Custom Fields", "inventory_custom_fields", (
        "id", "name", "field_type", "position", "created_at",
    )),
    ("Stock History", "inventory_audit", (
        "id", "actor", "item_name", "action", "old_quantity", "new_quantity",
        "note", "created_at",
    )),
    ("Events", "event_stock_events", (
        "id", "name", "columns", "rows", "stock_deducted", "status",
        "completed_at", "completion_rows", "created_at", "updated_at",
    )),
)

PAGE_SIZE = 500
MAX_CELL_TEXT_LENGTH = 30_000
_HEADER_FILL = PatternFill("solid", fgColor="29223A")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_TITLE_FONT = Font(bold=True, size=14, color="8B3DCC")


def fetch_all_table_rows(client: Any, table_name: str, page_size: int = PAGE_SIZE) -> list[dict[str, Any]]:
    """Fetch a whole table in stable ID order, beyond the API's per-request limit."""
    if page_size < 1:
        raise ValueError("page_size must be a positive integer")

    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        result = (
            client.table(table_name)
            .select("*")
            .order("id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = result.data or []
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def _excel_value(value: Any) -> Any:
    """Convert nested database values to readable Excel cells without evaluating text."""
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _sheet_columns(
    rows: Sequence[Mapping[str, Any]],
    defaults: Sequence[str],
) -> list[tuple[str, int]]:
    names = list(dict.fromkeys(str(name) for name in defaults))
    for row in rows:
        for name in row:
            text_name = str(name)
            if text_name not in names:
                names.append(text_name)

    layout: list[tuple[str, int]] = []
    for name in names:
        longest = max(
            (len(str(_excel_value(row.get(name)) or "")) for row in rows),
            default=0,
        )
        parts = max(1, math.ceil(longest / MAX_CELL_TEXT_LENGTH))
        layout.append((name, parts))
    return layout


def _append_text_cell(sheet: Any, row: int, column: int, value: Any) -> None:
    cell = sheet.cell(row=row, column=column, value=value)
    if isinstance(value, str):
        # Keep database text such as =1+1 as plain text when opened in Excel.
        cell.data_type = "s"


def build_export_workbook(
    table_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    exported_at: datetime | None = None,
) -> Workbook:
    """Create a multi-sheet workbook with all values represented as safe cells."""
    exported_at = exported_at or datetime.now(timezone.utc)
    exported_at_text = exported_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    defaults = {sheet: columns for sheet, _table, columns in EXPORT_TABLES}
    table_names = {sheet: table for sheet, table, _columns in EXPORT_TABLES}

    workbook = Workbook()
    info = workbook.active
    info.title = "Export Info"
    info.sheet_view.showGridLines = False
    info.append(["The Liquid Affair - Data Export"])
    info["A1"].font = _TITLE_FONT
    info.append(["Exported at (UTC)", exported_at_text])
    info.append([])
    info.append(["Worksheet", "Database table", "Rows"])
    for cell in info[4]:
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
    for sheet_name, rows in table_rows.items():
        info.append([sheet_name, table_names.get(sheet_name, ""), len(rows)])
    info.freeze_panes = "A5"
    info.column_dimensions["A"].width = 28
    info.column_dimensions["B"].width = 32
    info.column_dimensions["C"].width = 14

    for sheet_name, rows in table_rows.items():
        sheet = workbook.create_sheet(title=sheet_name[:31])
        sheet.sheet_view.showGridLines = False
        layout = _sheet_columns(rows, defaults.get(sheet_name, ()))
        headers: list[str] = []
        for name, parts in layout:
            if parts == 1:
                headers.append(name)
            else:
                headers.extend(f"{name} (part {part + 1})" for part in range(parts))
        if headers:
            sheet.append(headers)
            for cell in sheet[1]:
                cell.fill = _HEADER_FILL
                cell.font = _HEADER_FONT
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions

        for row_index, record in enumerate(rows, start=2):
            column_index = 1
            for name, parts in layout:
                value = _excel_value(record.get(name))
                if isinstance(value, str) and parts > 1:
                    pieces = [
                        value[offset:offset + MAX_CELL_TEXT_LENGTH]
                        for offset in range(0, len(value), MAX_CELL_TEXT_LENGTH)
                    ]
                    pieces.extend([None] * (parts - len(pieces)))
                else:
                    pieces = [value] + [None] * (parts - 1)
                for piece in pieces:
                    _append_text_cell(sheet, row_index, column_index, piece)
                    column_index += 1

        for column_index, header in enumerate(headers, start=1):
            sample_values = [
                sheet.cell(row=row_index, column=column_index).value
                for row_index in range(1, min(sheet.max_row, 101) + 1)
            ]
            width = max((len(str(value)) for value in sample_values if value is not None), default=12)
            sheet.column_dimensions[get_column_letter(column_index)].width = min(max(width + 2, 12), 42)

    return workbook


def create_excel_export_file(
    table_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    """Write a uniquely named fresh workbook for Gradio's download component."""
    now = datetime.now(timezone.utc)
    workbook = build_export_workbook(table_rows, exported_at=now)
    export_dir = Path(tempfile.gettempdir()) / "liquid_affair_excel_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    filename = f"liquid_affair_export_{now.strftime('%Y%m%d_%H%M%S_%f')}.xlsx"
    destination = export_dir / filename
    workbook.save(destination)
    return str(destination)
