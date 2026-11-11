"""Case-insensitive matching helpers for inventory labels."""


def normalized_name(value):
    """Normalize a name for matching while preserving its saved display form."""
    return str(value or "").strip().casefold()


def unique_labels(values):
    """Keep the first saved spelling while deduplicating labels by case."""
    labels = []
    seen = set()
    for value in values:
        label = str(value or "").strip()
        key = normalized_name(label)
        if key and key not in seen:
            labels.append(label)
            seen.add(key)
    return labels


def unique_item_names(records):
    """Return one display name for each case-insensitive inventory item name."""
    return sorted(unique_labels(record.get("name") for record in records), key=str.casefold)

