"""Helpers for presenting and validating event stock return quantities."""
from decimal import Decimal, InvalidOperation


def _whole_quantity(value, label, blank_is_zero=False):
    text = "" if value is None else str(value).strip()
    if not text:
        if blank_is_zero:
            return 0
        raise ValueError(f"Enter a return quantity for {label}; enter 0 if none is returned.")
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        raise ValueError(f"{label} must be a whole-number quantity.")
    if not number.is_finite() or number != number.to_integral_value() or number < 0:
        raise ValueError(f"{label} must be a non-negative whole-number quantity.")
    return int(number)


def aggregate_event_allocations(columns, rows):
    """Return each allocated item and its total in first-seen row order."""
    normalized_columns = [str(column or "").strip().casefold() for column in columns or []]
    if "item" not in normalized_columns or "quantity" not in normalized_columns:
        raise ValueError("The event sheet must keep its Item and Quantity columns.")
    item_index = normalized_columns.index("item")
    quantity_index = normalized_columns.index("quantity")
    totals = {}
    names = {}
    for row in rows or []:
        row = list(row or [])
        item = str(row[item_index] if item_index < len(row) else "").strip()
        raw_quantity = row[quantity_index] if quantity_index < len(row) else None
        if not item:
            if raw_quantity is not None and str(raw_quantity).strip():
                raise ValueError("Choose an inventory item for every quantity entered.")
            continue
        key = item.casefold()
        quantity = _whole_quantity(raw_quantity, f'Quantity for "{item}"', blank_is_zero=True)
        names.setdefault(key, item)
        totals[key] = totals.get(key, 0) + quantity
    return [[names[key], totals[key]] for key in names]


def build_completion_rows(allocations, return_values):
    """Build completion rows from UI data, trusting only server-side allocations."""
    if len(return_values or []) != len(allocations):
        raise ValueError("Enter a return quantity for every allocated item; enter 0 if none is returned.")
    completion_rows = []
    for (item_name, allocated), raw_row in zip(allocations, return_values or []):
        row = list(raw_row or [])
        if len(row) < 3:
            raise ValueError("The completion table is incomplete. Reopen it and review all allocated items.")
        submitted_item = str(row[0] or "").strip()
        submitted_allocation = _whole_quantity(row[1], f'Allocated quantity for "{item_name}"')
        allocated_quantity = _whole_quantity(allocated, f'Allocated quantity for "{item_name}"')
        if submitted_item.casefold() != str(item_name).strip().casefold() or submitted_allocation != allocated_quantity:
            raise ValueError("Allocated items cannot be changed. Reopen the completion dialog and try again.")
        returned_quantity = _whole_quantity(row[2], f'Return quantity for "{item_name}"')
        if returned_quantity > allocated_quantity:
            raise ValueError(f'Return quantity for "{item_name}" cannot exceed allocated quantity ({allocated_quantity}).')
        used_quantity = allocated_quantity - returned_quantity
        completion_rows.append([str(item_name).strip(), allocated_quantity, used_quantity, returned_quantity])
    return completion_rows
