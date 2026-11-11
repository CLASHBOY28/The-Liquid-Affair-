import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI
from supabase import Client, create_client
from chatbot_capabilities import calculate_math, current_india_time_context
from app_auth import load_auth_users
from audit_identity import pack_actor_note, unpack_actor_note
from inventory_matching import normalized_name, unique_item_names, unique_labels

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
APP_USERS = load_auth_users(os.environ)
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")

gemini_client = (
    OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    if GEMINI_API_KEY else None
)
groq_client = (
    OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    if GROQ_API_KEY else None
)
cloudflare_client = None
if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN:
    cloudflare_client = OpenAI(
        api_key=CLOUDFLARE_API_TOKEN,
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1/",
    )
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def gemini_chat_completion(**kwargs):
    """Use Gemini only when configured; try a short retry and configured fallback."""
    if gemini_client is None:
        raise RuntimeError("No AI provider is configured. Add GROQ_API_KEY or GEMINI_API_KEY.")
    requested_model = kwargs.get("model", MODEL)
    models = [requested_model]
    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
    if fallback_model and fallback_model != requested_model:
        models.append(fallback_model)

    last_error = None
    for model_index, model in enumerate(models):
        # One quick retry on the primary model; do not repeatedly retry 429 quota limits.
        max_attempts = 2 if model_index == 0 else 1
        for attempt in range(max_attempts):
            try:
                request = dict(kwargs)
                request["model"] = model
                return gemini_client.chat.completions.create(**request)
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code == 429:
                    break
                if exc.status_code not in (500, 502, 503, 504) or attempt + 1 >= max_attempts:
                    break
                time.sleep(1)
    raise last_error


def _provider_error_detail(exc):
    """Extract a short provider message without exposing request headers or API keys."""
    import re

    detail = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("detail") or "")
        else:
            detail = str(error)
    elif isinstance(body, str):
        detail = body

    if not detail:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    error = payload.get("error", payload)
                    detail = str(error.get("message") or error.get("detail") or error)
            except Exception:
                try:
                    detail = str(response.text or "")
                except Exception:
                    detail = ""

    if not detail:
        detail = str(exc)

    # Defensive redaction in case a provider echoes a credential-like value.
    detail = re.sub(
        r"(?i)(api[_ -]?key|authorization|bearer)\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        detail,
    )
    return " ".join(detail.split())[:320]


def ai_chat_completion(provider="Auto (recommended)", **kwargs):
    """Route requests to the selected provider, preserving provider error details."""
    request = dict(kwargs)
    request.pop("model", None)

    def call_groq():
        if groq_client is None:
            raise RuntimeError("Groq is not configured. Add GROQ_API_KEY in Railway Variables.")
        payload = dict(request)
        payload["model"] = GROQ_MODEL
        # GPT-OSS 120B supports function calling but not parallel tool calls.
        payload["parallel_tool_calls"] = False
        return groq_client.chat.completions.create(**payload)

    def call_gemini():
        if gemini_client is None:
            raise RuntimeError("Gemini is not configured. Add GEMINI_API_KEY in Railway Variables.")
        payload = dict(request)
        payload["model"] = MODEL
        # Gemini's OpenAI-compatible endpoint supports tools/tool_choice; omit
        # the optional parallel_tool_calls field for maximum compatibility.
        payload.pop("parallel_tool_calls", None)
        return gemini_chat_completion(**payload)

    def call_cloudflare():
        if cloudflare_client is None:
            raise RuntimeError(
                "Cloudflare Workers AI is not configured. Add CLOUDFLARE_ACCOUNT_ID "
                "and CLOUDFLARE_API_TOKEN in Railway Variables."
            )
        payload = dict(request)
        payload["model"] = CLOUDFLARE_MODEL
        payload.pop("parallel_tool_calls", None)
        return cloudflare_client.chat.completions.create(**payload)

    selected = (provider or "Auto (recommended)").strip()
    routes = {
        "Groq": call_groq,
        "Gemini": call_gemini,
        "Cloudflare Workers AI": call_cloudflare,
    }
    if selected in routes:
        return routes[selected]()

    # Prefer Groq, then Gemini, with Cloudflare as the final fallback.
    errors = []
    configured_routes = (
        ("Groq", call_groq, groq_client is not None),
        ("Gemini", call_gemini, gemini_client is not None),
        ("Cloudflare Workers AI", call_cloudflare, cloudflare_client is not None),
    )
    for name, fn, configured in configured_routes:
        if not configured:
            continue
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            detail = _provider_error_detail(exc)
            suffix = f" (HTTP {status})" if status else ""
            errors.append(f"{name}{suffix}: {detail}")
            logging.warning("AI provider %s failed%s: %s", name, suffix, detail)

    raise RuntimeError(
        "No configured AI provider could complete the request. "
        + ("; ".join(errors) if errors else
           "Add an API key for Groq, Gemini, or Cloudflare Workers AI.")
    )

def category_id_for(name: str, create: bool = False):
    display_name = (name or "").strip()
    key = normalized_name(display_name)
    if not key:
        raise ValueError("Enter a category name.")
    result = supabase.table("categories").select("id, name").execute()
    existing = next(
        (row for row in (result.data or []) if normalized_name(row.get("name")) == key),
        None,
    )
    if existing:
        return existing["id"]
    if not create:
        return None
    inserted = supabase.table("categories").insert({"name": display_name.lower()}).execute()
    if not inserted.data:
        raise RuntimeError("Could not create category.")
    return inserted.data[0]["id"]

def categories_list():
    result = supabase.table("categories").select("id, name").order("name").execute()
    return result.data or []


def custom_fields_list():
    try:
        result = (supabase.table("inventory_custom_fields")
                  .select("id, name, field_type, position")
                  .order("position").execute())
        return result.data or []
    except Exception:
        return []


def inventory_records():
    result = (supabase.table("inventory_items")
              .select("id, name, quantity, unit, reorder_threshold, expiry_date, custom_values, updated_at, categories(name)")
              .order("name").execute())
    return result.data or []


def find_item_by_name_and_category(name, category_id):
    """Look up an item without treating letter case as part of its identity."""
    result = (supabase.table("inventory_items").select("id, name, quantity")
              .eq("category_id", category_id).execute())
    key = normalized_name(name)
    matches = [
        row for row in (result.data or [])
        if normalized_name(row.get("name")) == key
    ]
    if len(matches) > 1:
        raise ValueError(
            "This category has multiple items that differ only by capitalization. "
            "Resolve the duplicate rows before changing stock."
        )
    return matches[0] if matches else None

def inventory_headers():
    return ["ID", "Item", "Category", "Quantity", "Type", "Reorder at", "Expiry date"] + [
        field["name"] for field in custom_fields_list()
    ] + ["Updated"]


def inventory_rows(low_only: bool = False):
    records = inventory_records()
    if low_only:
        records = [r for r in records if r["quantity"] <= r["reorder_threshold"]]
    fields = custom_fields_list()
    rows = []
    for item in records:
        category = item.get("categories") or {}
        custom = item.get("custom_values") or {}
        rows.append([
            item["id"], item["name"], category.get("name", "Uncategorised"),
            item["quantity"], item.get("unit") or "pcs",
            item["reorder_threshold"], item.get("expiry_date") or "",
            *[custom.get(field["name"], "") for field in fields],
            item.get("updated_at", "")
        ])
    return rows


def _actor_column_missing(exc):
    detail = str(exc).lower()
    return "actor" in detail and any(
        marker in detail for marker in ("column", "schema cache", "42703", "pgrst204")
    )


def audit_records(limit=200):
    table = supabase.table("inventory_audit")
    try:
        result = (table.select(
            "created_at, actor, item_name, action, old_quantity, new_quantity, note"
        ).order("created_at", desc=True).limit(limit).execute())
        records = result.data or []
    except Exception as exc:
        if not _actor_column_missing(exc):
            raise
        result = (table.select(
            "created_at, item_name, action, old_quantity, new_quantity, note"
        ).order("created_at", desc=True).limit(limit).execute())
        records = result.data or []
        for row in records:
            row["actor"], row["note"] = unpack_actor_note(row.get("note"))
    for row in records:
        row["actor"], row["note"] = unpack_actor_note(row.get("note"), row.get("actor"))
    return records


def audit_rows():
    return [[r.get("created_at", ""), r.get("actor", "Legacy"), r.get("item_name", ""),
             r.get("action", ""), r.get("old_quantity"), r.get("new_quantity"), r.get("note", "")]
            for r in audit_records()]


def write_audit(item_name, action, old_qty=None, new_qty=None, note=None, actor=None):
    payload = {
        "actor": actor or "Unknown", "item_name": item_name, "action": action,
        "old_quantity": old_qty, "new_quantity": new_qty, "note": note,
    }
    try:
        supabase.table("inventory_audit").insert(payload).execute()
    except Exception as exc:
        if not _actor_column_missing(exc):
            raise
        payload.pop("actor")
        payload["note"] = pack_actor_note(actor, note)
        supabase.table("inventory_audit").insert(payload).execute()


def authenticated_username(request):
    return (getattr(request, "username", None) or "Unknown") if request else "Unknown"


def suggest_category_for_item(item_name: str):
    """Suggest an existing category from saved items and conservative item-name keywords.

    Returns a category name only when the match is unambiguous; otherwise returns
    an empty string so the user can choose manually.
    """
    import re
    from difflib import SequenceMatcher

    name = normalized_name(item_name)
    if not name:
        return ""

    categories = categories_list()
    if not categories:
        return ""

    def words(value):
        return set(re.findall(r"[a-z0-9]+", normalized_name(value)))

    # Reuse the category of an item already in inventory when its name matches.
    records = inventory_records()
    exact = [
        (record.get("categories") or {}).get("name")
        for record in records
        if normalized_name(record.get("name")) == name
        and (record.get("categories") or {}).get("name")
    ]
    if len(set(exact)) == 1:
        return exact[0]

    # A close name match can carry over an existing item's category.
    best_score, best_category = 0.0, ""
    for record in records:
        existing_name = normalized_name(record.get("name"))
        existing_category = (record.get("categories") or {}).get("name", "")
        if not existing_name or not existing_category:
            continue
        score = SequenceMatcher(None, name, existing_name).ratio()
        new_words, old_words = words(name), words(existing_name)
        if new_words and old_words:
            overlap = len(new_words & old_words) / len(new_words | old_words)
            score = max(score, overlap)
        if score > best_score:
            best_score, best_category = score, existing_category
    if best_score >= 0.78:
        return best_category

    # Keyword groups are applied only when the corresponding category already exists.
    def compact(value):
        return "".join(re.findall(r"[a-z0-9]+", normalized_name(value)))

    # Match category names using normalized aliases, regardless of spaces,
    # ampersands, or singular/plural wording.
    category_aliases = {
        "juice": {"juice", "juices"},
        "use & throw": {"useandthrow", "usethrow", "usethrowaway", "disposable", "disposables"},
    }
    keyword_groups = {
        "juice": {
            "juice", "juices", "mango", "pineapple", "orange", "apple", "grape",
            "watermelon", "cranberry", "lychee", "passionfruit", "guava",
            "pomegranate", "mosambi", "lemon", "lime", "fruit",
        },
        "use & throw": {
            "tissue", "tissues", "straw", "straws", "napkin", "napkins",
            "disposable", "disposables", "paper", "cup", "cups", "plate",
            "plates", "spoon", "spoons", "fork", "forks", "garbage", "bin",
        },
    }
    normalized_categories = {
        compact(category["name"]): category["name"] for category in categories
    }

    matches = []
    item_words = words(name)
    for group, keywords in keyword_groups.items():
        if item_words & keywords:
            for alias in category_aliases[group]:
                if alias in normalized_categories:
                    matches.append(normalized_categories[alias])
                    break

    # If a category's own meaningful name appears in the item name, use it.
    # This supports any categories the user adds later without editing code.
    generic_words = {"and", "the", "for", "of", "item", "items"}
    for category in categories:
        category_words = words(category["name"]) - generic_words
        if category_words and category_words.issubset(item_words):
            matches.append(category["name"])

    matches = unique_labels(matches)
    return matches[0] if len(matches) == 1 else ""


def inventory_item_form_defaults(item_name):
    """Fill saved category/unit when an inventory item is selected by name."""
    name = (item_name or "").strip()
    if not name:
        return gr.update(value=None), gr.update()

    matches = [
        record for record in inventory_records()
        if normalized_name(record.get("name")) == normalized_name(name)
    ]
    if matches:
        categories = unique_labels(
            (record.get("categories") or {}).get("name") for record in matches
        )
        units = unique_labels(record.get("unit") or "pcs" for record in matches)

        if len(categories) == 1:
            category_update = gr.update(value=categories[0])
        elif len(categories) > 1:
            category_update = gr.update(
                value=None,
                info="This item name is in multiple categories. Select the right category.",
            )
        else:
            category_update = gr.update(value=None)

        if len(units) == 1:
            unit_update = gr.update(value=units[0])
        else:
            unit_update = gr.update(
                info="This item name has different types in multiple categories. Confirm the type.",
            )
        return category_update, unit_update

    category = suggest_category_for_item(name)
    return gr.update(value=category or None), gr.update()

def save_item(name, category, quantity, threshold, unit, expiry_date=None, actor=None):
    name = (name or "").strip()
    category = (category or "").strip()
    unit = (unit or "pcs").strip() or "pcs"
    if name and not category:
        category = suggest_category_for_item(name)
    if not name or not category:
        raise ValueError("I couldn't confidently identify a category for this new item. Please tell me its category once.")
    if quantity is None or int(quantity) < 0 or threshold is None or int(threshold) < 0:
        raise ValueError("Quantity and reorder threshold must be zero or greater.")
    quantity, threshold = int(quantity), int(threshold)
    expiry_date = (expiry_date or "").strip() or None
    if expiry_date:
        datetime.strptime(expiry_date, "%Y-%m-%d")
    cat_id = category_id_for(category, create=True)
    found = find_item_by_name_and_category(name, cat_id)
    if found:
        name = found["name"]
        old = found["quantity"]
        (supabase.table("inventory_items").update({
            "quantity": quantity, "reorder_threshold": threshold,
            "unit": unit, "expiry_date": expiry_date,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", found["id"]).execute())
        write_audit(name, "set_quantity", old, quantity, "Set from inventory form", actor)
        return f"Updated {name}: {old} → {quantity} {unit}."
    (supabase.table("inventory_items").insert({
        "name": name, "category_id": cat_id, "quantity": quantity,
        "reorder_threshold": threshold, "unit": unit, "expiry_date": expiry_date
    }).execute())
    write_audit(name, "create", None, quantity, "Added from inventory form", actor)
    return f"Added {name}: {quantity} {unit}."

def add_custom_field(name, field_type="text", actor=None):
    name = (name or "").strip()
    if not name:
        raise ValueError("Enter a column name.")
    reserved = {h.casefold() for h in ["ID", "Item", "Category", "Quantity", "Type", "Unit", "Reorder at", "Expiry date", "Updated"]}
    if name.casefold() in reserved:
        raise ValueError("That name is already used by a standard inventory column.")
    existing = [f["name"].casefold() for f in custom_fields_list()]
    if name.casefold() in existing:
        raise ValueError("That custom column already exists.")
    if field_type not in ("text", "number", "date"):
        field_type = "text"
    supabase.table("inventory_custom_fields").insert({
        "name": name, "field_type": field_type,
        "position": len(existing) + 1
    }).execute()
    write_audit(f"Column: {name}", "add_column", note=f"Added {field_type} inventory field", actor=actor)
    return f"Added the '{name}' column."


def move_custom_field(name, direction, actor=None):
    fields = custom_fields_list()
    index = next((i for i, field in enumerate(fields) if normalized_name(field["name"]) == normalized_name(name)), None)
    if index is None:
        raise ValueError("Choose a custom column.")
    other = index - 1 if direction == "up" else index + 1
    if other < 0 or other >= len(fields):
        return "That column is already at the edge."
    first, second = fields[index], fields[other]
    supabase.table("inventory_custom_fields").update({"position": second["position"]}).eq("id", first["id"]).execute()
    supabase.table("inventory_custom_fields").update({"position": first["position"]}).eq("id", second["id"]).execute()
    write_audit(f"Column: {name}", "move_column", note=f"Moved {direction}", actor=actor)
    return f"Moved '{name}' {direction}."


def remove_custom_field(name, actor=None):
    field = next((f for f in custom_fields_list() if normalized_name(f["name"]) == normalized_name(name)), None)
    if not field:
        raise ValueError("Choose a custom column.")
    supabase.table("inventory_custom_fields").delete().eq("id", field["id"]).execute()
    write_audit(f"Column: {name}", "remove_column", note="Removed inventory field", actor=actor)
    return f"Removed '{name}' from the visible inventory columns. Existing saved values remain in the records."


def save_inventory_table(data, actor=None):
    if data is None:
        return "No table changes to save."
    headers = inventory_headers()
    fields = custom_fields_list()
    if hasattr(data, "values"):
        data = data.values.tolist()
    updated = 0
    for row in data:
        if not row or row[0] in (None, ""):
            continue
        item_id = int(row[0])
        custom_values = {}
        for idx, field in enumerate(fields, start=7):
            value = row[idx] if idx < len(row) else None
            if value not in (None, ""):
                custom_values[field["name"]] = value
        expiry = row[6] if len(row) > 6 and row[6] not in ("", None) else None
        if expiry:
            expiry = str(expiry)[:10]
            datetime.strptime(expiry, "%Y-%m-%d")
        values = {
            "quantity": int(row[3] or 0),
            "unit": str(row[4] or "pcs"),
            "reorder_threshold": int(row[5] or 0),
            "expiry_date": expiry,
            "custom_values": custom_values,
        }
        current_result = (supabase.table("inventory_items")
                          .select("name, quantity, unit, reorder_threshold, expiry_date, custom_values")
                          .eq("id", item_id).limit(1).execute())
        if not current_result.data:
            continue
        current = current_result.data[0]
        changed = any(current.get(key) != value for key, value in values.items())
        if not changed:
            continue
        (supabase.table("inventory_items").update({
            **values, "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", item_id).execute())
        write_audit(
            current.get("name") or "Inventory item", "table_update",
            current.get("quantity"), values["quantity"],
            "Updated from inventory table", actor,
        )
        updated += 1
    return f"Saved changes for {updated} inventory item(s)."


def adjust_item(name, category, delta, actor=None):
    name, category = (name or "").strip(), (category or "").strip()
    if not name or delta is None:
        raise ValueError("Enter the item and stock change.")
    if not category:
        category = suggest_category_for_item(name)
    if not category:
        raise ValueError(f"I couldn't identify a unique category for {name}. Please select its category.")
    cat_id = category_id_for(category, create=False)
    if cat_id is None:
        raise ValueError(f"Category '{category}' does not exist.")
    found = find_item_by_name_and_category(name, cat_id)
    if not found:
        raise ValueError(f"{name} was not found in {category}.")
    name = found["name"]
    old = found["quantity"]
    new = old + int(delta)
    if new < 0:
        raise ValueError(f"Not enough stock. Current quantity is {old}.")
    (supabase.table("inventory_items").update({
        "quantity": new, "updated_at": datetime.now(timezone.utc).isoformat()
    }).eq("id", found["id"]).execute())
    write_audit(name, "adjust", old, new, f"delta={int(delta)}", actor)
    return f"{name}: {old} → {new}."

def add_category(name, actor=None):
    existing = category_id_for(name, create=False)
    category_id_for(name, create=True)
    if existing is None:
        write_audit(f"Category: {name.strip()}", "add_category", note="Created category", actor=actor)
    return f"Category '{name.strip().lower()}' is ready."


def dashboard_summary():
    records = inventory_records()
    low = [r for r in records if r["quantity"] <= r["reorder_threshold"]]
    units = len(records)
    total = sum(r["quantity"] for r in records)
    category_count = len(categories_list())
    return (
        '<div class="tla-kpi-grid">'
        '<div class="tla-kpi"><span>Inventory items</span><strong>' + str(units) + '</strong></div>'
        '<div class="tla-kpi"><span>Total units</span><strong>' + str(total) + '</strong></div>'
        '<div class="tla-kpi"><span>Low-stock alerts</span><strong>' + str(len(low)) + '</strong></div>'
        '<div class="tla-kpi"><span>Categories</span><strong>' + str(category_count) + '</strong></div>'
        '</div><p class="tla-kpi-note">Low-stock alerts include items at or below their reorder threshold.</p>'
    )


TOOLS = [
    {"type": "function", "function": {"name": "calculate_math",
     "description": "Accurately calculate arithmetic. Use for all numerical calculations rather than mental arithmetic. Supports parentheses and +, -, *, /, //, %, and **. Pass only a mathematical expression, e.g. (250*3)+18 or 500*0.18.",
     "parameters": {"type": "object", "properties": {
         "expression": {"type": "string", "description": "A mathematical expression using numbers and supported arithmetic operators."}},
         "required": ["expression"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "add_or_update_item",
     "description": "Add an inventory item or set its exact quantity. Category is optional: infer it from the existing item saved in inventory. Never ask for category when the item name matches one existing inventory item.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"}, "category": {"type": "string", "description": "Optional. For existing items, omit this and use the saved category."},
         "quantity": {"type": "integer", "minimum": 0},
         "reorder_threshold": {"type": "integer", "minimum": 0},
         "unit": {"type": "string"},
         "expiry_date": {"type": "string", "description": "Optional expiry date in YYYY-MM-DD format"}},
         "required": ["name", "quantity"]}}},
    {"type": "function", "function": {"name": "adjust_stock",
     "description": "Increase or decrease an existing item's stock by a delta. Category is optional; infer it from saved inventory. Never ask for category when the item name matches one existing inventory item. Ask only if multiple saved items share that name.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"}, "category": {"type": "string", "description": "Optional. Omit for an existing item; use its saved category."},
         "delta": {"type": "integer"}},
         "required": ["name", "delta"]}}},
    {"type": "function", "function": {"name": "list_inventory",
     "description": "List all inventory items.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "low_stock",
     "description": "List items at or below their reorder threshold.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "add_category",
     "description": "Create an inventory category.", "parameters": {
         "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "list_categories",
     "description": "List all inventory categories.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def execute_tool(name: str, args: dict[str, Any], actor=None) -> str:
    try:
        if name == "calculate_math":
            return calculate_math(args["expression"])
        if name == "add_or_update_item":
            return save_item(args["name"], args.get("category"), args["quantity"],
                             args.get("reorder_threshold", 5), args.get("unit", "pcs"),
                             args.get("expiry_date"), actor=actor)
        if name == "adjust_stock":
            return adjust_item(args["name"], args.get("category"), args["delta"], actor)
        if name == "list_inventory":
            return json.dumps(inventory_records(), default=str)
        if name == "low_stock":
            return json.dumps([r for r in inventory_records()
                               if r["quantity"] <= r["reorder_threshold"]], default=str)
        if name == "add_category":
            return add_category(args["name"], actor)
        if name == "list_categories":
            return json.dumps(categories_list(), default=str)
        return "Unknown inventory action."
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def history_to_messages(history):
    """Normalize Gradio's messages and legacy tuple history into API messages."""
    messages = []

    def text_content(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(part for part in parts if part)
        return ""

    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = text_content(turn.get("content"))
            if role in ("user", "assistant") and content.strip():
                messages.append({"role": role, "content": content})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_content = text_content(turn[0])
            assistant_content = text_content(turn[1])
            if user_content.strip():
                messages.append({"role": "user", "content": user_content})
            if assistant_content.strip():
                messages.append({"role": "assistant", "content": assistant_content})
    return messages[-24:]


SYSTEM_PROMPT = """
You have access to a live clock context appended to your system instructions on every request.
For every arithmetic calculation (including percentages, discounts, totals, ratios, and multi-step maths),
call the calculate_math tool with a valid expression and report its result. Never guess or do arithmetic
mentally when the calculator tool can be used. For percentage-of calculations, convert to multiplication
(e.g. 18% of 500 -> 500*18/100). For a percentage increase/decrease, calculate the exact expression.
If a word problem needs multiple steps, use the calculator for each step or combine them in one expression.
You are the friendly, capable AI assistant for The Liquid Affair, a mobile
bartending and bar-operations business.

CHAT LIKE A GENERAL-PURPOSE AI ASSISTANT:
- Handle greetings, normal conversation, explanations, brainstorming, writing,
  calculations, recipes, event planning, and general questions naturally.
- Do not force every conversation into inventory management.
- Understand informal wording, typos, shorthand, and follow-up messages using
  the conversation history. Ask one clear question when essential information
  is missing.
- Be concise by default, but explain thoroughly when asked.
- Never claim to have checked live information unless a tool actually checked it.

INVENTORY AND BUSINESS DATA:
- Use the supplied tools whenever the user asks to look up or change live inventory.
- Never invent inventory quantities, categories, expiry dates, or successful writes.
- “Add/set stock to N” means set the exact quantity to N. “Increase/decrease by N”
  means adjust the current quantity by that amount.
- For an existing inventory item, its saved category is the source of truth. If the user gives the item name and quantity or stock change, do NOT ask for its category; call the tool with item name and quantity/delta only and let the app infer the category.
- Ask for category only when the item is new and no confident category can be inferred, or when the same item name exists in multiple categories.
- Explain tool results in clear, natural language. Do not expose raw JSON to the user.
- If a tool returns an error, explain that the operation did not complete.
- Never perform an inventory write merely because the user is discussing an item;
  the user must clearly ask for an action.
"""


def _chat_error_message(exc):
    """Return a useful provider error without exposing credentials."""
    status = getattr(exc, "status_code", None)
    detail = _provider_error_detail(exc)

    if status == 429:
        return "The AI provider reached its rate or usage limit. " + (detail or "Please try again later.")
    if status in (401, 403):
        return "The AI provider rejected its API key or the key lacks permission. Check the key in Railway Variables. " + detail
    if status == 404:
        return "The selected AI model or endpoint was not found. Check the model setting. " + detail
    if status in (400, 422):
        return "The AI provider rejected the request format. Provider detail: " + detail
    if status in (500, 502, 503, 504):
        return "The AI provider is temporarily unavailable. " + detail
    if status:
        return f"The AI request failed (HTTP {status}). " + detail

    logging.exception("AI assistant request failed")
    message = str(exc).strip()
    if message:
        return "The AI assistant could not complete the request. " + _provider_error_detail(exc)
    return "I couldn't reach the AI service just now. Please try again in a moment."

def _quick_inventory_reply(user_text, actor=None):
    """Handle common inventory commands directly and safely, without an AI round-trip."""
    import re

    raw = (user_text or "").strip()
    normalized = " ".join(raw.lower().split()).strip(" .!?")

    low_stock_phrases = (
        "show low stock", "show me low stock", "show all low stock",
        "list low stock", "low stock items", "show low stock items",
        "show items low on stock", "what is low on stock",
        "what's low on stock", "what needs reordering", "show reorder list",
        "list items to reorder",
    )
    if any(phrase in normalized for phrase in low_stock_phrases):
        records = inventory_records()
        low_items = [
            item for item in records
            if int(item.get("quantity") or 0) <= int(item.get("reorder_threshold") or 0)
        ]
        if not low_items:
            return "You're all set — no items are currently at or below their reorder threshold."
        lines = [f"### Low-stock items ({len(low_items)})"]
        for item in low_items:
            category_data = item.get("categories") or {}
            if isinstance(category_data, list):
                category_data = category_data[0] if category_data else {}
            category = category_data.get("name", "Uncategorised") if isinstance(category_data, dict) else "Uncategorised"
            lines.append(
                f"- **{item.get('name', 'Unnamed item')}** — "
                f"{item.get('quantity', 0)} {item.get('unit') or 'pcs'} available; "
                f"reorder at {item.get('reorder_threshold', 0)} ({category})."
            )
        return "\n".join(lines)

    if normalized in ("list inventory", "show inventory", "show all inventory",
                      "list all inventory", "display inventory", "inventory list"):
        records = inventory_records()
        if not records:
            return "Your inventory is currently empty."
        lines = [f"### Inventory ({len(records)} items)"]
        for item in records:
            category_data = item.get("categories") or {}
            if isinstance(category_data, list):
                category_data = category_data[0] if category_data else {}
            category = category_data.get("name", "Uncategorised") if isinstance(category_data, dict) else "Uncategorised"
            lines.append(
                f"- **{item.get('name', 'Unnamed item')}** — "
                f"{item.get('quantity', 0)} {item.get('unit') or 'pcs'} "
                f"({category}; reorder at {item.get('reorder_threshold', 0)})"
            )
        return "\n".join(lines)

    # Explicit category creation: "Add category Syrups".
    category_match = re.fullmatch(
        r"(?:add|create)\\s+(?:a\\s+)?category\\s+(.+)",
        raw, flags=re.IGNORECASE
    )
    if category_match:
        category_name = category_match.group(1).strip().strip("\\\"' ")
        if not category_name:
            return "Please provide a category name."
        try:
            return add_category(category_name, actor)
        except Exception as exc:
            return f"Could not add category '{category_name}': {exc}"

    # Stock adjustment with or without a category:
    # "Reduce Straws by 5" / "Reduce Straws by 5 in Disposables".
    match = re.fullmatch(
        r"(reduce|decrease|subtract|deduct|increase|add)\\s+(.+?)\\s+by\\s+(\\d+)(?:\\s+in\\s+(.+))?",
        raw, flags=re.IGNORECASE
    )
    if match:
        verb, item_name, amount, category = match.groups()
        delta = int(amount) * (-1 if verb.lower() in ("reduce", "decrease", "subtract", "deduct") else 1)
        try:
            return adjust_item(item_name.strip(), category.strip() if category else None, delta, actor)
        except Exception as exc:
            return f"Stock was not changed: {exc}"

    return None

def _object_value(value, key, default=None):
    """Read a field from either an SDK model or a plain dictionary."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def chat_fn(message, history, provider="Auto (recommended)", request: gr.Request = None):
    """General AI chat with selectable provider and bounded tool execution."""
    user_text = (message or "").strip()
    if not user_text:
        return "What would you like help with?"

    actor = authenticated_username(request)
    quick_reply = _quick_inventory_reply(user_text, actor)
    if quick_reply is not None:
        return quick_reply

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + current_india_time_context()}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": user_text})

    max_tool_rounds = max(1, min(int(os.getenv("CHAT_MAX_TOOL_ROUNDS", "4")), 6))

    for _round in range(max_tool_rounds + 1):
        try:
            response = ai_chat_completion(
                provider=provider,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                parallel_tool_calls=False,
            )
        except APIStatusError as exc:
            return _chat_error_message(exc)
        except Exception as exc:
            logging.exception("Unexpected AI assistant error: %s", type(exc).__name__)
            return _chat_error_message(exc)

        if not getattr(response, "choices", None):
            return "The AI service returned an empty response. Please try again."

        assistant_message = response.choices[0].message
        tool_calls = (
            _object_value(assistant_message, "tool_calls")
            or _object_value(assistant_message, "toolCalls")
            or []
        )
        answer = _object_value(assistant_message, "content")
        if not tool_calls:
            return answer.strip() if isinstance(answer, str) and answer.strip() else (
                "I couldn't generate a response. Please try asking in a different way."
            )

        if _round >= max_tool_rounds:
            return (
                "I need another step to finish this request, but I reached the safe "
                "tool-call limit. Please review the latest inventory state before "
                "asking me to continue, so an action is not accidentally repeated."
            )

        # Normalize SDK model objects and Gemini's camelCase toolCalls alias into
        # the snake_case OpenAI message format before sending the next tool result.
        normalized_calls = []
        for index, call in enumerate(tool_calls):
            function = _object_value(call, "function", {}) or {}
            call_id = str(_object_value(call, "id") or f"call_{_round}_{index}")
            function_name = str(_object_value(function, "name", "") or "")
            raw_arguments = _object_value(function, "arguments", "{}") or "{}"
            if not isinstance(raw_arguments, str):
                raw_arguments = json.dumps(raw_arguments)
            normalized_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": raw_arguments,
                },
            })

        assistant_payload = {"role": "assistant"}
        if answer is not None:
            assistant_payload["content"] = answer
        assistant_payload["tool_calls"] = normalized_calls
        messages.append(assistant_payload)

        for call in normalized_calls:
            tool_name = call["function"]["name"]
            call_id = call["id"]
            try:
                raw_arguments = call["function"]["arguments"] or "{}"
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                result = execute_tool(tool_name, arguments, actor)
            except Exception as exc:
                logging.exception("Assistant tool execution failed: %s", tool_name)
                result = json.dumps({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(result),
            })

    return (
        "I reached the limit for this request while working with the inventory. "
        "No further actions were run. Please ask me to continue."
    )

def refresh_all():
    items = inventory_records()
    cats = categories_list()
    return (
        dashboard_summary(),
        gr.update(value=inventory_rows(), headers=inventory_headers()),
        gr.update(value=inventory_rows(low_only=True), headers=inventory_headers()),
        [[c["id"], c["name"]] for c in cats],
        audit_rows(),
        gr.update(choices=unique_item_names(items)),
        gr.update(choices=[c["name"] for c in cats]),
        gr.update(choices=unique_item_names(items)),
        gr.update(choices=[c["name"] for c in cats]),
        gr.update(choices=[f["name"] for f in custom_fields_list()]),
        mobile_inventory_cards(),
        mobile_inventory_cards(low_only=True),
        mobile_category_cards(),
        mobile_audit_cards(),
    )


def add_item_ui(name, category, quantity, threshold, unit, expiry_date, request: gr.Request = None):
    try:
        message = save_item(name, category, quantity, threshold, unit, expiry_date,
                            actor=authenticated_username(request))
        return message, dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()
    except Exception as exc:
        return f"Could not save item: {exc}", dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()



def suggest_adjust_category(item_name):
    """Auto-select the item's category when its name identifies one category."""
    name = normalized_name(item_name)
    if not name:
        return gr.update(value=None)
    matches = [
        record for record in inventory_records()
        if normalized_name(record.get("name")) == name
    ]
    category_names = unique_labels(
        (record.get("categories") or {}).get("name") for record in matches
    )
    if len(category_names) == 1:
        return gr.update(value=category_names[0])
    if len(category_names) > 1:
        return gr.update(value=None, info="This item name exists in multiple categories. Please select the correct category.")
    return gr.update(value=None)

def adjust_ui(name, category, delta, request: gr.Request = None):
    try:
        message = adjust_item(name, category, delta, authenticated_username(request))
        return message, dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()
    except Exception as exc:
        return f"Could not adjust stock: {exc}", dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()


def custom_field_ui(name, field_type, request: gr.Request = None):
    try:
        message = add_custom_field(name, field_type, authenticated_username(request))
        return message, gr.update(headers=inventory_headers(), value=inventory_rows()), gr.update(headers=inventory_headers(), value=inventory_rows(True)), gr.update(choices=[f["name"] for f in custom_fields_list()])
    except Exception as exc:
        return f"Could not add column: {exc}", gr.update(), gr.update(), gr.update()


def move_field_ui(name, direction, request: gr.Request = None):
    try:
        message = move_custom_field(name, direction, authenticated_username(request))
        return message, gr.update(headers=inventory_headers(), value=inventory_rows()), gr.update(headers=inventory_headers(), value=inventory_rows(True))
    except Exception as exc:
        return f"Could not move column: {exc}", gr.update(), gr.update()


def remove_field_ui(name, request: gr.Request = None):
    try:
        message = remove_custom_field(name, authenticated_username(request))
        return message, gr.update(headers=inventory_headers(), value=inventory_rows()), gr.update(headers=inventory_headers(), value=inventory_rows(True)), gr.update(choices=[f["name"] for f in custom_fields_list()])
    except Exception as exc:
        return f"Could not remove column: {exc}", gr.update(), gr.update(), gr.update()


def save_table_ui(data, request: gr.Request = None):
    try:
        message = save_inventory_table(data, authenticated_username(request))
        return message, gr.update(value=inventory_rows()), gr.update(value=inventory_rows(True)), dashboard_summary(), audit_rows()
    except Exception as exc:
        return f"Could not save table: {exc}", gr.update(), gr.update(), dashboard_summary(), audit_rows()


def add_category_ui(name, request: gr.Request = None):
    try:
        message = add_category(name, authenticated_username(request))
        cats = categories_list()
        return message, [[c["id"], c["name"]] for c in cats], gr.update(choices=[c["name"] for c in cats])
    except Exception as exc:
        return f"Could not add category: {exc}", [[c["id"], c["name"]] for c in categories_list()], gr.update()



def _html_escape(value):
    """Escape database values before placing them in mobile HTML cards."""
    from html import escape
    return escape("" if value is None else str(value))


def mobile_inventory_cards(low_only=False):
    """Compact, touch-friendly inventory cards for narrow Android screens."""
    records = inventory_records()
    if low_only:
        records = [r for r in records if r["quantity"] <= r["reorder_threshold"]]
    if not records:
        return '<div class="mobile-empty">No inventory items to show.</div>'

    cards = []
    for item in records:
        category = (item.get("categories") or {}).get("name") or "Uncategorised"
        quantity = item.get("quantity", 0)
        threshold = item.get("reorder_threshold", 0)
        low = quantity <= threshold
        status = "Low stock" if low else "In stock"
        expiry = item.get("expiry_date") or "No expiry date"
        cards.append(
            '<article class="tla-mobile-card">'
            '<div class="tla-card-top"><div><h3>' + _html_escape(item.get("name")) +
            '</h3><p>' + _html_escape(category) + '</p></div>'
            '<span class="tla-stock-pill ' + ("low" if low else "ok") + '">' + status + '</span></div>'
            '<div class="tla-card-metrics"><div><small>Available</small><strong>' +
            _html_escape(quantity) + ' ' + _html_escape(item.get("unit") or "pcs") +
            '</strong></div><div><small>Reorder at</small><strong>' +
            _html_escape(threshold) + '</strong></div></div>'
            '<p class="tla-card-foot">Expiry: ' + _html_escape(expiry) + '</p>'
            '</article>'
        )
    return '<div class="tla-mobile-card-list">' + "".join(cards) + '</div>'


def mobile_category_cards():
    categories = categories_list()
    if not categories:
        return '<div class="mobile-empty">No categories created yet.</div>'
    return '<div class="tla-mobile-card-list">' + "".join(
        '<article class="tla-mobile-card tla-category-card"><strong>' +
        _html_escape(category.get("name")) + '</strong></article>'
        for category in categories
    ) + '</div>'


def mobile_audit_cards():
    records = audit_records(limit=100)
    if not records:
        return '<div class="mobile-empty">No stock changes recorded yet.</div>'
    cards = []
    for row in records:
        created = row.get("created_at") or ""
        try:
            created = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone().strftime("%d %b %Y · %I:%M %p")
        except Exception:
            pass
        cards.append(
            '<article class="tla-mobile-card"><div class="tla-card-top"><div><h3>' +
            _html_escape(row.get("item_name") or "Inventory item") + '</h3><p>' +
            _html_escape(row.get("action")) + '</p></div><span class="tla-audit-date">' +
            _html_escape(created) + '</span></div><div class="tla-card-metrics"><div><small>Previous</small><strong>' +
            _html_escape(row.get("old_quantity") if row.get("old_quantity") is not None else "—") +
            '</strong></div><div><small>New quantity</small><strong>' +
            _html_escape(row.get("new_quantity") if row.get("new_quantity") is not None else "—") +
            '</strong></div></div><p class="tla-card-foot"><strong>User:</strong> ' +
            _html_escape(row.get("actor") or "Legacy") + '<br>' +
            _html_escape(row.get("note") or "") + '</p></article>'
        )
    return '<div class="tla-mobile-card-list">' + "".join(cards) + '</div>'


def theme_css(theme):
    palettes = {
        "Neon Lounge": ("#090812", "#171326", "#f5c96a", "#d94dff", "#f5f0ff", "#27d9d0"),
        "Tropical Luxe": ("#071b19", "#12352e", "#e8c77b", "#43c7a5", "#f2f7e9", "#b7dc75"),
        "Classic Gold": ("#11100e", "#242019", "#d8b66a", "#a98a52", "#f5f0e6", "#c9a96e"),
    }
    bg, panel, gold, accent, ink, glow = palettes.get(theme, palettes["Neon Lounge"])
    return f"""
    <style>
      :root {{ --tla-bg:{bg}; --tla-panel:{panel}; --tla-gold:{gold}; --tla-accent:{accent}; --tla-ink:{ink}; --tla-glow:{glow}; }}
      body, .gradio-container {{ background:var(--tla-bg)!important; color:var(--tla-ink)!important; }}
      .gradio-container .block, .gradio-container .tabitem, .gradio-container .panel {{ background:var(--tla-panel)!important; border-color:color-mix(in srgb,var(--tla-gold) 45%,transparent)!important; }}
      .gradio-container h1, .gradio-container h2, .gradio-container h3 {{ color:var(--tla-gold)!important; }}
      .gradio-container button.primary {{ background:var(--tla-accent)!important; border-color:var(--tla-gold)!important; }}
      .tla-header {{ display:flex; align-items:center; justify-content:center; gap:22px; padding:22px 12px; margin-bottom:12px; border:1px solid color-mix(in srgb,var(--tla-gold) 55%,transparent); border-radius:18px; background:radial-gradient(ellipse at center,color-mix(in srgb,var(--tla-accent) 17%,var(--tla-panel)),var(--tla-bg)); box-shadow:0 0 28px color-mix(in srgb,var(--tla-glow) 18%,transparent); }}
      .tla-logo {{ width:132px; height:132px; flex:none; filter:drop-shadow(0 0 8px var(--tla-accent)); }}
      .tla-brand {{ text-align:left; }}
      .tla-brand h1 {{ margin:0; font-size:clamp(27px,5vw,43px); letter-spacing:2px; color:var(--tla-gold); }}
      .tla-brand p {{ margin:8px 0 0; letter-spacing:4px; font-size:12px; color:var(--tla-ink); opacity:.85; }}
      @media(max-width:520px) {{ .tla-header {{ gap:10px; padding:14px 8px; }} .tla-logo {{ width:94px;height:94px; }} .tla-brand p {{letter-spacing:2px;font-size:10px;}} }}
    </style>
    """


def brand_header():
    return """
    <div class="tla-header">
      <svg class="tla-logo" viewBox="0 0 200 200" role="img" aria-label="The Liquid Affair neon cocktail logo">
        <defs>
          <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#ed4bd8"/><stop offset=".48" stop-color="#ffbd58"/><stop offset="1" stop-color="#35d8dc"/></linearGradient>
          <linearGradient id="liquid" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#e85cff"/><stop offset="1" stop-color="#ffbd58"/></linearGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <circle cx="100" cy="100" r="91" fill="#090812" stroke="url(#ring)" stroke-width="4" filter="url(#glow)"/>
        <text x="39" y="58" fill="#ffd47b" font-size="17" font-family="cursive" font-style="italic">The</text>
        <text x="20" y="99" fill="#e96aff" font-size="35" font-family="cursive" font-style="italic" filter="url(#glow)">Liquid</text>
        <text x="23" y="135" fill="#ffd06b" font-size="34" font-family="cursive" font-style="italic" filter="url(#glow)">Affair</text>
        <path d="M133 45 L177 45 L156 83 L156 119 M142 119 H170 M155 84 L173 55" fill="none" stroke="#43dce0" stroke-width="3" stroke-linecap="round"/>
        <path d="M137 55 Q156 75 174 55 L166 72 Q156 84 146 72Z" fill="url(#liquid)" opacity=".9"/>
        <path d="M151 43 Q171 25 178 43" fill="none" stroke="#ffcf68" stroke-width="3" stroke-linecap="round"/>
        <circle cx="178" cy="40" r="5" fill="#ed58d5"/>
      </svg>
      <div class="tla-brand"><h1>THE LIQUID AFFAIR</h1><p>PREMIUM BARTENDING SERVICES</p></div>
    </div>
    """


APP_CSS = """
:root { color-scheme: dark; }
.theme-style-host { display:none !important; }
.tla-app-shell { width:100% !important; min-width:0 !important; gap:12px !important; }
.tla-mobile-nav { display:flex !important; flex-flow:row nowrap !important; align-items:stretch !important; gap:6px !important; width:100% !important; min-width:0 !important; }
.tla-nav-button { flex:1 1 0 !important; min-width:0 !important; width:0 !important; padding:9px 3px !important; font-size:clamp(9px,2.65vw,13px) !important; line-height:1.15 !important; white-space:normal !important; overflow-wrap:anywhere !important; }
.tla-page { width:100% !important; min-width:0 !important; }
@media(max-width:760px) {
  .tla-mobile-nav { display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important; gap:5px !important; }
  .tla-nav-button { min-height:48px !important; padding:8px 2px !important; font-size:clamp(9px,2.7vw,12px) !important; }
}
html, body { width:100%; min-width:320px; margin:0; overflow-x:hidden; }
.gradio-container { width:100% !important; max-width:1440px !important; margin:0 auto !important; padding:clamp(12px,2vw,28px) !important; box-sizing:border-box !important; background:var(--tla-bg,#090812) !important; }
.gradio-container .block, .gradio-container .tabitem, .gradio-container .panel { min-width:0 !important; border-radius:16px !important; }
.gradio-container .row { min-width:0 !important; align-items:stretch !important; }
.gradio-container .column { min-width:0 !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select { min-height:44px !important; font-size:16px !important; border-radius:11px !important; }
.gradio-container button { min-height:44px !important; border-radius:11px !important; font-weight:650 !important; touch-action:manipulation; }
.tla-header { display:flex !important; align-items:center !important; justify-content:flex-start !important; gap:18px !important; padding:18px 22px !important; margin:0 0 14px !important; border:1px solid color-mix(in srgb,var(--tla-gold) 35%,transparent) !important; border-radius:18px !important; background:linear-gradient(110deg,color-mix(in srgb,var(--tla-panel) 95%,transparent),var(--tla-bg)) !important; box-shadow:0 8px 26px rgba(0,0,0,.16) !important; }
.tla-logo { width:86px !important; height:86px !important; flex:none !important; }
.tla-brand h1 { margin:0 !important; font-size:clamp(23px,3vw,34px) !important; letter-spacing:1.5px !important; line-height:1.15 !important; }
.tla-brand p { margin:6px 0 0 !important; font-size:10px !important; letter-spacing:2.2px !important; opacity:.78 !important; }
.mobile-only { display:none !important; }
.tla-mobile-card-list { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr)); gap:12px; width:100%; }
.tla-mobile-card { min-width:0; padding:16px; border-radius:15px; background:linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.018)); border:1px solid color-mix(in srgb,var(--tla-gold,#d8b66a) 28%,transparent); box-shadow:0 5px 18px rgba(0,0,0,.12); }
.tla-card-top { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
.tla-card-top h3 { font-size:16px; line-height:1.25; margin:0 0 4px; overflow-wrap:anywhere; }
.tla-card-top p,.tla-card-foot { font-size:12px; opacity:.72; margin:0; }
.tla-stock-pill { font-size:11px; font-weight:750; border-radius:999px; padding:6px 10px; white-space:nowrap; }
.tla-stock-pill.low { background:rgba(255,91,91,.15); color:#ff9b9b; }
.tla-stock-pill.ok { background:rgba(57,205,143,.13); color:#80e8b8; }
.tla-card-metrics { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:13px 0 10px; }
.tla-card-metrics div { min-width:0; background:rgba(0,0,0,.16); border-radius:11px; padding:10px; }
.tla-card-metrics small { display:block; opacity:.68; font-size:11px; margin-bottom:4px; }
.tla-card-metrics strong { font-size:16px; overflow-wrap:anywhere; }
.tla-card-foot { padding-top:9px; border-top:1px solid rgba(255,255,255,.08); overflow-wrap:anywhere; }
.tla-audit-date { font-size:10px; opacity:.7; text-align:right; max-width:120px; }
.mobile-empty { padding:20px; border:1px dashed rgba(255,255,255,.2); border-radius:14px; opacity:.7; text-align:center; }
.tla-category-card { padding:16px; }
.tla-kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:10px 0 8px; }
.tla-kpi { min-width:0; padding:15px; border-radius:14px; background:color-mix(in srgb,var(--tla-panel) 90%,#fff 3%); border:1px solid color-mix(in srgb,var(--tla-gold) 25%,transparent); }
.tla-kpi span { display:block; font-size:12px; opacity:.72; margin-bottom:7px; }
.tla-kpi strong { display:block; font-size:clamp(20px,2vw,27px); color:var(--tla-gold); line-height:1.1; overflow-wrap:anywhere; }
.tla-kpi-note { font-size:12px; opacity:.68; margin:8px 2px 0; }
@media(max-width:760px) {
  html, body { width:100% !important; min-width:320px !important; overflow-x:hidden !important; }
  .gradio-container { padding:10px 12px calc(22px + env(safe-area-inset-bottom)) !important; gap:12px !important; }
  .gradio-container .row { flex-direction:row !important; flex-wrap:wrap !important; gap:10px !important; min-width:0 !important; }
  .gradio-container .row > * { min-width:min(100%,145px) !important; flex:1 1 145px !important; }
  .gradio-container .tab-nav, .gradio-container [role="tablist"] { display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important; align-items:center !important; justify-content:flex-start !important; gap:7px !important; width:100% !important; max-width:100% !important; height:auto !important; overflow-x:auto !important; overflow-y:hidden !important; white-space:nowrap !important; padding:5px 2px 11px !important; margin:0 0 10px !important; scrollbar-width:none !important; -webkit-overflow-scrolling:touch !important; background:var(--tla-bg,#090812) !important; }
  .gradio-container .tab-nav::-webkit-scrollbar, .gradio-container [role="tablist"]::-webkit-scrollbar { display:none !important; }
  .gradio-container .tab-nav button, .gradio-container [role="tablist"] button, .gradio-container button[role="tab"] { display:inline-flex !important; visibility:visible !important; opacity:1 !important; flex:0 0 auto !important; width:auto !important; max-width:none !important; min-width:max-content !important; min-height:43px !important; padding:10px 14px !important; margin:0 !important; font-size:13px !important; line-height:1.15 !important; white-space:nowrap !important; }
  /* Remove Gradio's overflow/ellipsis tab menu; show the actual section tabs. */
  .gradio-container .tab-nav button[aria-label*="more"],
  .gradio-container .tab-nav button[aria-label*="More"],
  .gradio-container [role="tablist"] button[aria-label*="more"],
  .gradio-container [role="tablist"] button[aria-label*="More"],
  .gradio-container .tab-nav .more,
  .gradio-container .tab-nav [data-testid*="more"],
  .gradio-container [role="tablist"] [data-testid*="more"] { display:none !important; }
  .gradio-container .tab-nav { border-bottom:1px solid rgba(255,255,255,.12) !important; }
  .gradio-container .tab-nav button[role="tab"],
  .gradio-container [role="tablist"] button[role="tab"] { border-radius:10px 10px 0 0 !important; font-weight:700 !important; }
  .gradio-container .tab-nav button.selected,
  .gradio-container [role="tablist"] button[aria-selected="true"] { color:#f39a52 !important; border-bottom:2px solid #f39a52 !important; }
  .tla-header { gap:12px !important; padding:12px 14px !important; margin-bottom:10px !important; border-radius:15px !important; }
  .tla-logo { width:62px !important; height:62px !important; }
  .tla-brand h1 { font-size:clamp(18px,5.2vw,24px) !important; letter-spacing:1px !important; }
  .tla-brand p { font-size:8px !important; letter-spacing:1.2px !important; margin-top:4px !important; }
  .gradio-container h1 { font-size:23px !important; }
  .gradio-container h2 { font-size:20px !important; }
  .gradio-container h3 { font-size:17px !important; }
  .gradio-container .block { padding:12px !important; border-radius:14px !important; }
  .gradio-container button { min-height:44px !important; }
  .gradio-container input, .gradio-container textarea, .gradio-container select { min-height:44px !important; font-size:16px !important; }
  .gradio-container .tabitem { padding:12px !important; }
  .desktop-only { display:none !important; }
  .mobile-only { display:block !important; width:100% !important; min-width:0 !important; }
  .tla-mobile-card-list { grid-template-columns:minmax(0,1fr) !important; gap:10px !important; }
  .tla-mobile-card { padding:14px !important; }
  .tla-card-metrics { gap:8px !important; margin:11px 0 9px !important; }
  .tla-card-metrics div { padding:9px 8px !important; }
  .tla-card-metrics strong { font-size:15px !important; }
  .tla-kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)) !important; gap:8px !important; }
  .tla-kpi { padding:12px !important; }
  .tla-kpi span { font-size:11px !important; }
  .tla-kpi strong { font-size:22px !important; }
  .gradio-container .chatbot { height:min(62vh,620px) !important; min-height:360px !important; }
  .gradio-container .message-wrap { font-size:15px !important; }
  .gradio-container footer { padding-bottom:env(safe-area-inset-bottom) !important; }
}
@media(max-width:390px) { .gradio-container { padding-left:9px !important; padding-right:9px !important; } .tla-kpi { padding:10px !important; } .tla-kpi strong { font-size:20px !important; } }

/* Android layout correction: full-width controls, no cramped 145px columns. */

@media (max-width: 760px) {
  html, body { width:100% !important; min-width:320px !important; overflow-x:hidden !important; }
  .gradio-container {
    width:100% !important; max-width:100% !important;
    padding:12px 14px calc(24px + env(safe-area-inset-bottom)) !important;
    margin:0 !important; gap:14px !important; box-sizing:border-box !important;
  }
  /* Every form row becomes a clean vertical stack. Avoid narrow side-by-side fields. */
  .gradio-container .row {
    display:flex !important; flex-direction:column !important; flex-wrap:nowrap !important;
    align-items:stretch !important; gap:12px !important; width:100% !important;
    min-width:0 !important; margin-left:0 !important; margin-right:0 !important;
  }
  .gradio-container .row > *,
  .gradio-container .row > .form,
  .gradio-container .row > .container,
  .gradio-container .row > .column {
    display:block !important; flex:0 0 auto !important;
    width:100% !important; min-width:0 !important; max-width:100% !important;
    box-sizing:border-box !important;
  }
  .gradio-container .row .wrap,
  .gradio-container .row .container,
  .gradio-container .row .form,
  .gradio-container .row .block {
    min-width:0 !important; max-width:100% !important;
    box-sizing:border-box !important;
  }
  /* Preserve the real sections in one horizontally scrollable nav strip. */
  .gradio-container .tab-nav,
  .gradio-container [role="tablist"] {
    display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important;
    justify-content:flex-start !important; align-items:center !important; gap:8px !important;
    width:100% !important; max-width:100% !important; height:auto !important;
    overflow-x:auto !important; overflow-y:hidden !important; white-space:nowrap !important;
    padding:4px 2px 12px !important; margin:0 0 12px !important;
    scrollbar-width:none !important; -webkit-overflow-scrolling:touch !important;
  }
  .gradio-container .tab-nav::-webkit-scrollbar,
  .gradio-container [role="tablist"]::-webkit-scrollbar { display:none !important; }
  .gradio-container .tab-nav button,
  .gradio-container [role="tablist"] button,
  .gradio-container button[role="tab"] {
    display:inline-flex !important; flex:0 0 auto !important;
    width:auto !important; min-width:max-content !important; max-width:none !important;
    min-height:44px !important; padding:10px 15px !important;
    font-size:13px !important; white-space:nowrap !important;
  }
  .tla-header { width:100% !important; gap:12px !important; padding:14px !important; margin:0 0 12px !important; }
  .tla-logo { width:64px !important; height:64px !important; }
  .tla-brand { min-width:0 !important; }
  .tla-brand h1 { font-size:clamp(19px,5.5vw,25px) !important; letter-spacing:1px !important; overflow-wrap:anywhere !important; }
  .tla-brand p { font-size:8px !important; letter-spacing:1.1px !important; }
  .gradio-container .block { width:100% !important; min-width:0 !important; padding:13px !important; border-radius:15px !important; }
  .gradio-container label { font-size:13px !important; line-height:1.25 !important; }
  .gradio-container input,
  .gradio-container textarea,
  .gradio-container select,
  .gradio-container [role="combobox"] {
    width:100% !important; min-width:0 !important; max-width:100% !important;
    min-height:48px !important; font-size:16px !important; box-sizing:border-box !important;
  }
  .gradio-container .wrap,
  .gradio-container .form,
  .gradio-container .container { min-width:0 !important; max-width:100% !important; }
  .gradio-container button { min-height:46px !important; }
  .gradio-container .primary,
  .gradio-container .stop { width:100% !important; }
  .gradio-container .tabitem { padding:12px 4px !important; min-width:0 !important; }
  .mobile-only { display:block !important; width:100% !important; min-width:0 !important; }
  .desktop-only { display:none !important; }
  .tla-mobile-card-list { display:grid !important; grid-template-columns:minmax(0,1fr) !important; gap:10px !important; width:100% !important; }
  .tla-mobile-card { width:100% !important; min-width:0 !important; padding:14px !important; box-sizing:border-box !important; }
  .tla-card-metrics { grid-template-columns:repeat(2,minmax(0,1fr)) !important; gap:9px !important; }
  .tla-card-metrics div { min-width:0 !important; }
  .tla-card-metrics strong { font-size:15px !important; overflow-wrap:anywhere !important; }
  .tla-kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)) !important; gap:9px !important; }
  .tla-kpi { min-width:0 !important; padding:12px !important; }
  .tla-kpi strong { font-size:22px !important; }
  .gradio-container .chatbot { height:min(62vh,620px) !important; min-height:360px !important; }
  .gradio-container .message-wrap { font-size:15px !important; }
  .gradio-container footer { padding-bottom:env(safe-area-inset-bottom) !important; }
}
@media (max-width: 390px) {
  .gradio-container { padding-left:10px !important; padding-right:10px !important; }
  .tla-header { padding:11px !important; }
  .tla-logo { width:56px !important; height:56px !important; }
  .tla-kpi { padding:10px !important; }
  .tla-kpi strong { font-size:20px !important; }
}



/* Keep the complete tab strip wider than its viewport so Gradio does not
   collapse remaining tabs into its responsive three-dot menu. */
@media (max-width: 760px) {
  .gradio-container .tabs,
  .gradio-container [class*="tabs"] {
    min-width:0 !important;
  }
  .gradio-container .tab-nav,
  .gradio-container [role="tablist"] {
    display:flex !important;
    flex-direction:row !important;
    flex-wrap:nowrap !important;
    justify-content:flex-start !important;
    align-items:stretch !important;
    width:max-content !important;
    min-width:max-content !important;
    max-width:none !important;
    overflow:visible !important;
    gap:clamp(8px,2.5vw,14px) !important;
    padding:4px 2px 9px !important;
    margin:0 !important;
    white-space:nowrap !important;
  }
  /* Make the tab strip itself horizontally scrollable rather than
     allowing Gradio to move overflow tabs into its menu. */
  .gradio-container .tab-nav {
    scrollbar-width:none !important;
    -webkit-overflow-scrolling:touch !important;
  }
  .gradio-container .tab-nav::-webkit-scrollbar { display:none !important; }
  .gradio-container .tab-nav button[role="tab"],
  .gradio-container [role="tablist"] button[role="tab"],
  .gradio-container button[role="tab"] {
    display:inline-flex !important;
    flex:0 0 auto !important;
    width:auto !important;
    min-width:max-content !important;
    max-width:none !important;
    min-height:44px !important;
    padding:9px 5px !important;
    margin:0 !important;
    font-size:clamp(11px,3vw,14px) !important;
    line-height:1.15 !important;
    white-space:nowrap !important;
    border-radius:0 !important;
    background:transparent !important;
  }
  /* Hide only the generated overflow-menu trigger. */
  .gradio-container .tab-nav button[aria-label*="More"],
  .gradio-container .tab-nav button[aria-label*="more"],
  .gradio-container [role="tablist"] button[aria-label*="More"],
  .gradio-container [role="tablist"] button[aria-label*="more"],
  .gradio-container .tab-nav [aria-haspopup="menu"],
  .gradio-container [role="tablist"] [aria-haspopup="menu"] {
    display:none !important;
  }
}
"""


with gr.Blocks(title="The Liquid Affair | Inventory") as demo:
    theme_styles = gr.HTML(theme_css("Neon Lounge"), elem_classes=["theme-style-host"])
    gr.HTML(brand_header())
    with gr.Row():
        theme_selector = gr.Dropdown(
            choices=["Neon Lounge", "Tropical Luxe", "Classic Gold"],
            value="Neon Lounge", label="Choose your theme", scale=1
        )
    theme_selector.change(fn=theme_css, inputs=theme_selector, outputs=theme_styles)
    gr.Markdown("### Your bar, your stock — all in one place")
    gr.Markdown("Manage inventory, stock levels and expiry dates from one place.", elem_classes=["tla-subtitle"])
    with gr.Column(elem_classes=["tla-app-shell"]):
        with gr.Row(elem_classes=["tla-mobile-nav"]):
            nav_dashboard = gr.Button("Dashboard", variant="primary", elem_classes=["tla-nav-button"])
            nav_inventory = gr.Button("Inventory", variant="secondary", elem_classes=["tla-nav-button"])
            nav_categories = gr.Button("Categories", variant="secondary", elem_classes=["tla-nav-button"])
            nav_ai = gr.Button("AI Assistant", variant="secondary", elem_classes=["tla-nav-button"])
            nav_history = gr.Button("Stock History", variant="secondary", elem_classes=["tla-nav-button"])
        with gr.Column(visible=True, elem_classes=["tla-page"]) as dashboard_page:
            summary = gr.Markdown()
            refresh_button = gr.Button("Refresh dashboard")
            gr.Markdown("#### Low-stock items")
            low_table = gr.Dataframe(
                headers=["ID", "Item", "Category", "Quantity", "Type", "Reorder at", "Expiry date", "Updated"],
                interactive=False, wrap=False, elem_classes=["desktop-only"]
            )
            low_stock_mobile = gr.HTML(value=mobile_inventory_cards(low_only=True), elem_classes=["mobile-only"])
        with gr.Column(visible=False, elem_classes=["tla-page"]) as inventory_page:
            gr.Markdown("#### Add item or set its exact stock")
            with gr.Row():
                item_name = gr.Dropdown(
                    label="Item name", choices=[], allow_custom_value=True, filterable=True,
                    info="Choose an existing item or type a new name.",
                )
                item_category = gr.Dropdown(label="Category", choices=[], allow_custom_value=True, info="Auto-suggests a category as you type; you can change it.")
            with gr.Row():
                item_quantity = gr.Number(
                    label="Exact quantity", value=0, precision=0,
                    info="The total amount currently in stock for this item type.",
                )
                item_unit = gr.Textbox(
                    label="Type", value="bottles",
                    info="Enter how this item is counted, such as packs, bottles, or pieces.",
                )
                item_threshold = gr.Number(
                    label="Reorder threshold", value=5, precision=0,
                    info="Shows a low-stock alert when the quantity reaches or falls below this number.",
                )
            item_name.change(
                fn=inventory_item_form_defaults,
                inputs=item_name,
                outputs=[item_category, item_unit],
            )
            item_expiry = gr.Textbox(label="Expiry date (YYYY-MM-DD, optional)", placeholder="2027-06-30")
            save_button = gr.Button("Save item", variant="primary")
            save_status = gr.Markdown()
            gr.Markdown("#### Adjust existing stock (+/-)")
            gr.Markdown(
                "Choose the item and category, then enter a positive number to add stock or a negative number to record stock used. "
                "For example, +5 adds five and -2 removes two."
            )
            with gr.Row():
                adjust_name = gr.Dropdown(label="Select existing item", choices=[], interactive=True)
                adjust_category = gr.Dropdown(label="Select category", choices=[], interactive=True)
                adjust_name.change(
                    fn=suggest_adjust_category,
                    inputs=adjust_name,
                    outputs=adjust_category,
                )
                adjust_delta = gr.Number(
                    label="Stock change (+/-)", precision=0,
                    info="Use + to add received stock and - to subtract used or lost stock.",
                )
            adjust_button = gr.Button("Adjust stock")
            adjust_status = gr.Markdown()
            gr.Markdown("#### All inventory")
            gr.Markdown("Edit quantities, expiry dates, and custom values directly in the table, then save.", elem_classes=["desktop-only"])
            inventory_table = gr.Dataframe(
                headers=inventory_headers(), interactive=True, wrap=False, elem_classes=["desktop-only"]
            )
            inventory_mobile = gr.HTML(value=mobile_inventory_cards(), elem_classes=["mobile-only"])
            gr.Markdown("Use the forms above to edit items and stock. The full editable table is available on larger screens.", elem_classes=["desktop-only"])
            save_table_button = gr.Button("Save table changes", variant="primary", elem_classes=["desktop-only"])
            table_status = gr.Markdown(elem_classes=["desktop-only"])
        with gr.Column(visible=False, elem_classes=["tla-page"]) as categories_page:
            gr.Markdown("### Manage categories")
            category_name = gr.Textbox(label="New category", placeholder="Juices, Syrups, Glassware...")
            category_button = gr.Button("Add category", variant="primary")
            category_status = gr.Markdown()
            category_table = gr.Dataframe(headers=["ID", "Category"], interactive=False, elem_classes=["desktop-only"])
            category_mobile = gr.HTML(value=mobile_category_cards(), elem_classes=["mobile-only"])
            with gr.Accordion("Customize inventory columns", open=False):
                gr.Markdown("Add extra fields to your inventory, then move or remove them here.")
                custom_field_name = gr.Textbox(label="New column name", placeholder="Supplier")
                custom_field_type = gr.Dropdown(label="Value type", choices=["text", "number", "date"], value="text")
                add_field_button = gr.Button("Add column", variant="primary")
                field_status = gr.Markdown()
                manage_field = gr.Dropdown(label="Custom column", choices=[f["name"] for f in custom_fields_list()])
                move_direction = gr.Radio(choices=["up", "down"], value="up", label="Move")
                move_field_button = gr.Button("Move column")
                remove_field_button = gr.Button("Remove column", variant="stop")
        with gr.Column(visible=False, elem_classes=["tla-page"]) as ai_page:
            with gr.Row(equal_height=True):
                gr.Markdown("### 🤖 AI Assistant", scale=2)
                provider_selector = gr.Dropdown(
                    choices=[
                        "Auto (recommended)",
                        "Groq",
                        "Gemini",
                        "Cloudflare Workers AI",
                    ],
                    value="Auto (recommended)",
                    label="⚙️ AI Engine",
                    info="Choose which AI provider powers this chat.",
                    scale=2,
                    min_width=190,
                )
            gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[provider_selector],
                title="💬 Ask The Liquid Affair",
                description="Chat naturally and ask the assistant to work across your app.",
                examples=[
                    ["Show low stock", "Auto (recommended)"],
                    ["List inventory", "Auto (recommended)"],
                    ["Add category Syrups", "Auto (recommended)"],
                    ["Reduce Straws by 5 in Straws", "Auto (recommended)"],
                ],
            )
        with gr.Column(visible=False, elem_classes=["tla-page"]) as history_page:
            audit_table = gr.Dataframe(
                headers=["Date", "User", "Item", "Action", "Old quantity", "New quantity", "Note"],
                interactive=False, wrap=True, elem_classes=["desktop-only"]
            )
            audit_mobile = gr.HTML(value=mobile_audit_cards(), elem_classes=["mobile-only"])

    page_columns = [dashboard_page, inventory_page, categories_page, ai_page, history_page]
    page_buttons = [nav_dashboard, nav_inventory, nav_categories, nav_ai, nav_history]

    def select_page(selected):
        names = ["Dashboard", "Inventory", "Categories", "AI Assistant", "Stock History"]
        visibility = [gr.update(visible=(index == selected)) for index in range(5)]
        styles = [gr.update(variant=("primary" if index == selected else "secondary")) for index in range(5)]
        return visibility + styles

    for index, nav_button in enumerate(page_buttons):
        nav_button.click(
            fn=lambda index=index: select_page(index),
            outputs=page_columns + page_buttons,
            queue=False,
        )

    refresh_outputs = [summary, inventory_table, low_table, category_table, audit_table, item_name, item_category, adjust_name, adjust_category, manage_field, inventory_mobile, low_stock_mobile, category_mobile, audit_mobile]
    demo.load(refresh_all, outputs=refresh_outputs)
    refresh_button.click(refresh_all, outputs=refresh_outputs)
    save_button.click(
        add_item_ui,
        inputs=[item_name, item_category, item_quantity, item_threshold, item_unit, item_expiry],
        outputs=[save_status, summary, inventory_table, low_table, audit_table],
    ).then(refresh_all, outputs=refresh_outputs)
    adjust_button.click(
        adjust_ui, inputs=[adjust_name, adjust_category, adjust_delta],
        outputs=[adjust_status, summary, inventory_table, low_table, audit_table],
    ).then(refresh_all, outputs=refresh_outputs)
    save_table_button.click(
        save_table_ui, inputs=[inventory_table],
        outputs=[table_status, inventory_table, low_table, summary, audit_table],
    ).then(refresh_all, outputs=refresh_outputs)
    add_field_button.click(
        custom_field_ui, inputs=[custom_field_name, custom_field_type],
        outputs=[field_status, inventory_table, low_table, manage_field],
    ).then(refresh_all, outputs=refresh_outputs)
    move_field_button.click(
        move_field_ui, inputs=[manage_field, move_direction],
        outputs=[field_status, inventory_table, low_table],
    ).then(refresh_all, outputs=refresh_outputs)
    remove_field_button.click(
        remove_field_ui, inputs=[manage_field],
        outputs=[field_status, inventory_table, low_table, manage_field],
    ).then(refresh_all, outputs=refresh_outputs)
    category_button.click(
        add_category_ui, inputs=[category_name],
        outputs=[category_status, category_table, item_category],
    ).then(refresh_all, outputs=refresh_outputs)


APP_CSS += """
/* Override generic mobile .row stacking rules specifically for the app navigation. */
@media (max-width: 760px) {
  .gradio-container .row.tla-mobile-nav {
    display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important;
    align-items:stretch !important; gap:4px !important; width:100% !important;
    min-width:0 !important; margin:0 0 12px !important;
  }
  .gradio-container .row.tla-mobile-nav > * {
    display:flex !important; flex:1 1 0 !important; width:0 !important;
    min-width:0 !important; max-width:none !important; margin:0 !important;
  }
  .gradio-container .row.tla-mobile-nav button {
    width:100% !important; min-width:0 !important; padding:8px 2px !important;
    min-height:48px !important; font-size:clamp(9px,2.6vw,12px) !important;
    line-height:1.1 !important; white-space:normal !important; overflow-wrap:anywhere !important;
  }
}
"""

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        auth=APP_USERS,
        css=APP_CSS,
    )

# Mobile navigation layout update: tabs remain visible on narrow screens.