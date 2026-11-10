import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
APP_USERNAME = os.environ.get("APP_USERNAME", "")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")

if not APP_USERNAME or not APP_PASSWORD:
    raise RuntimeError("Set APP_USERNAME and APP_PASSWORD before starting the app.")

gemini_client = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
cloudflare_client = None
if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN:
    cloudflare_client = OpenAI(
        api_key=CLOUDFLARE_API_TOKEN,
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1/",
    )
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def gemini_chat_completion(**kwargs):
    """Use a short retry for temporary outages, then try the fallback model."""
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


def ai_chat_completion(**kwargs):
    """Prefer Cloudflare Workers AI when configured; fall back to Gemini on any provider error."""
    if cloudflare_client is not None:
        try:
            request = dict(kwargs)
            request["model"] = CLOUDFLARE_MODEL
            return cloudflare_client.chat.completions.create(**request)
        except Exception:
            # Gemini remains available if Cloudflare is unavailable, rate-limited, or misconfigured.
            pass
    gemini_request = dict(kwargs)
    gemini_request["model"] = MODEL
    return gemini_chat_completion(**gemini_request)


def category_id_for(name: str, create: bool = False):
    name = name.strip().lower()
    if not name:
        raise ValueError("Enter a category name.")
    result = supabase.table("categories").select("id").eq("name", name).limit(1).execute()
    if result.data:
        return result.data[0]["id"]
    if not create:
        return None
    inserted = supabase.table("categories").insert({"name": name}).execute()
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


def inventory_headers():
    return ["ID", "Item", "Category", "Quantity", "Unit", "Reorder at", "Expiry date"] + [
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


def audit_rows():
    result = (supabase.table("inventory_audit")
              .select("created_at, item_name, action, old_quantity, new_quantity, note")
              .order("created_at", desc=True).limit(200).execute())
    return [[r.get("created_at", ""), r.get("item_name", ""), r.get("action", ""),
             r.get("old_quantity"), r.get("new_quantity"), r.get("note", "")]
            for r in (result.data or [])]


def write_audit(item_name, action, old_qty=None, new_qty=None, note=None):
    supabase.table("inventory_audit").insert({
        "item_name": item_name, "action": action,
        "old_quantity": old_qty, "new_quantity": new_qty, "note": note,
    }).execute()


def save_item(name, category, quantity, threshold, unit, expiry_date=None):
    name = (name or "").strip()
    category = (category or "").strip()
    unit = (unit or "pcs").strip() or "pcs"
    if not name or not category:
        raise ValueError("Enter an item name and choose a category.")
    if quantity is None or int(quantity) < 0 or threshold is None or int(threshold) < 0:
        raise ValueError("Quantity and reorder threshold must be zero or greater.")
    quantity, threshold = int(quantity), int(threshold)
    expiry_date = (expiry_date or "").strip() or None
    if expiry_date:
        datetime.strptime(expiry_date, "%Y-%m-%d")
    cat_id = category_id_for(category, create=True)
    found = (supabase.table("inventory_items").select("id, quantity")
             .eq("name", name).eq("category_id", cat_id).limit(1).execute())
    if found.data:
        old = found.data[0]["quantity"]
        (supabase.table("inventory_items").update({
            "quantity": quantity, "reorder_threshold": threshold,
            "unit": unit, "expiry_date": expiry_date,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", found.data[0]["id"]).execute())
        write_audit(name, "set_quantity", old, quantity, "Dashboard")
        return f"Updated {name}: {old} → {quantity} {unit}."
    (supabase.table("inventory_items").insert({
        "name": name, "category_id": cat_id, "quantity": quantity,
        "reorder_threshold": threshold, "unit": unit, "expiry_date": expiry_date
    }).execute())
    write_audit(name, "create", None, quantity, "Dashboard")
    return f"Added {name}: {quantity} {unit}."


def add_custom_field(name, field_type="text"):
    name = (name or "").strip()
    if not name:
        raise ValueError("Enter a column name.")
    reserved = {h.lower() for h in ["ID", "Item", "Category", "Quantity", "Unit", "Reorder at", "Expiry date", "Updated"]}
    if name.lower() in reserved:
        raise ValueError("That name is already used by a standard inventory column.")
    existing = [f["name"].lower() for f in custom_fields_list()]
    if name.lower() in existing:
        raise ValueError("That custom column already exists.")
    if field_type not in ("text", "number", "date"):
        field_type = "text"
    supabase.table("inventory_custom_fields").insert({
        "name": name, "field_type": field_type,
        "position": len(existing) + 1
    }).execute()
    return f"Added the '{name}' column."


def move_custom_field(name, direction):
    fields = custom_fields_list()
    index = next((i for i, field in enumerate(fields) if field["name"] == name), None)
    if index is None:
        raise ValueError("Choose a custom column.")
    other = index - 1 if direction == "up" else index + 1
    if other < 0 or other >= len(fields):
        return "That column is already at the edge."
    first, second = fields[index], fields[other]
    supabase.table("inventory_custom_fields").update({"position": second["position"]}).eq("id", first["id"]).execute()
    supabase.table("inventory_custom_fields").update({"position": first["position"]}).eq("id", second["id"]).execute()
    return f"Moved '{name}' {direction}."


def remove_custom_field(name):
    field = next((f for f in custom_fields_list() if f["name"] == name), None)
    if not field:
        raise ValueError("Choose a custom column.")
    supabase.table("inventory_custom_fields").delete().eq("id", field["id"]).execute()
    return f"Removed '{name}' from the visible inventory columns. Existing saved values remain in the records."


def save_inventory_table(data):
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
        supabase.table("inventory_items").update({
            "quantity": int(row[3] or 0),
            "unit": str(row[4] or "pcs"),
            "reorder_threshold": int(row[5] or 0),
            "expiry_date": expiry,
            "custom_values": custom_values,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", item_id).execute()
        updated += 1
    return f"Saved changes for {updated} inventory item(s)."


def adjust_item(name, category, delta):
    name, category = (name or "").strip(), (category or "").strip()
    if not name or not category or delta is None:
        raise ValueError("Enter the item, category, and stock change.")
    cat_id = category_id_for(category, create=False)
    if cat_id is None:
        raise ValueError(f"Category '{category}' does not exist.")
    found = (supabase.table("inventory_items").select("id, quantity")
             .eq("name", name).eq("category_id", cat_id).limit(1).execute())
    if not found.data:
        raise ValueError(f"{name} was not found in {category}.")
    old = found.data[0]["quantity"]
    new = old + int(delta)
    if new < 0:
        raise ValueError(f"Not enough stock. Current quantity is {old}.")
    (supabase.table("inventory_items").update({
        "quantity": new, "updated_at": datetime.now(timezone.utc).isoformat()
    }).eq("id", found.data[0]["id"]).execute())
    write_audit(name, "adjust", old, new, f"delta={int(delta)}")
    return f"{name}: {old} → {new}."


def add_category(name):
    category_id_for(name, create=True)
    return f"Category '{name.strip().lower()}' is ready."


def dashboard_summary():
    records = inventory_records()
    low = [r for r in records if r["quantity"] <= r["reorder_threshold"]]
    units = len(records)
    total = sum(r["quantity"] for r in records)
    return (
        f"### Inventory overview\n"
        f"- **Items:** {units}\n"
        f"- **Total units (mixed units):** {total}\n"
        f"- **Low-stock alerts:** {len(low)}\n"
        f"- **Categories:** {len(categories_list())}\n\n"
        f"Low-stock count includes items at or below their reorder threshold."
    )


TOOLS = [
    {"type": "function", "function": {"name": "add_or_update_item",
     "description": "Add an inventory item or set its exact quantity. Creates the category if needed.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"}, "category": {"type": "string"},
         "quantity": {"type": "integer", "minimum": 0},
         "reorder_threshold": {"type": "integer", "minimum": 0},
         "unit": {"type": "string"},
         "expiry_date": {"type": "string", "description": "Optional expiry date in YYYY-MM-DD format"}},
         "required": ["name", "category", "quantity"]}}},
    {"type": "function", "function": {"name": "adjust_stock",
     "description": "Increase or decrease an existing item's stock by a delta.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"}, "category": {"type": "string"},
         "delta": {"type": "integer"}},
         "required": ["name", "category", "delta"]}}},
    {"type": "function", "function": {"name": "list_inventory",
     "description": "List all inventory items.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "low_stock",
     "description": "List items at or below their reorder threshold.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "add_category",
     "description": "Create an inventory category.", "parameters": {
         "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "list_categories",
     "description": "List all inventory categories.", "parameters": {"type": "object", "properties": {}}}},
]


def execute_tool(name: str, args: dict[str, Any]) -> str:
    try:
        if name == "add_or_update_item":
            return save_item(args["name"], args["category"], args["quantity"],
                             args.get("reorder_threshold", 5), args.get("unit", "pcs"),
                             args.get("expiry_date"))
        if name == "adjust_stock":
            return adjust_item(args["name"], args["category"], args["delta"])
        if name == "list_inventory":
            return json.dumps(inventory_records(), default=str)
        if name == "low_stock":
            return json.dumps([r for r in inventory_records()
                               if r["quantity"] <= r["reorder_threshold"]], default=str)
        if name == "add_category":
            return add_category(args["name"])
        if name == "list_categories":
            return json.dumps(categories_list(), default=str)
        return "Unknown inventory action."
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def history_to_messages(history):
    messages = []
    for turn in history or []:
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            if turn[0]:
                messages.append({"role": "user", "content": str(turn[0])})
            if turn[1]:
                messages.append({"role": "assistant", "content": str(turn[1])})
    return messages


SYSTEM_PROMPT = """
You are the inventory assistant for The Liquid Affair, a mobile bar business.
Use the supplied tools for every inventory lookup or change. Never invent stock values.
'Add/set stock to N' means set the exact quantity. 'Increase/decrease by N' means adjust
the current quantity. Ask one concise question when the item or category is ambiguous.
Report tool errors honestly. Do not claim an operation succeeded if its tool returned ERROR.
"""


def chat_fn(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": message})
    try:
        response = gemini_chat_completion(
            model=MODEL, messages=messages, tools=TOOLS,
            tool_choice="auto", parallel_tool_calls=False,
        )
    except APIStatusError as exc:
        if exc.status_code in (429, 500, 502, 503, 504):
            return "Gemini is temporarily unavailable (HTTP " + str(exc.status_code) + "). Please try again in a minute."
        return "Gemini API request failed (HTTP " + str(exc.status_code) + "). Please contact the app administrator."
    assistant_message = response.choices[0].message
    if not assistant_message.tool_calls:
        return assistant_message.content or "I couldn't generate a response."
    messages.append(assistant_message)
    for call in assistant_message.tool_calls:
        try:
            args = json.loads(call.function.arguments or "{}")
            result = execute_tool(call.function.name, args)
        except Exception as exc:
            result = f"ERROR: {type(exc).__name__}: {exc}"
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    try:
        follow_up = gemini_chat_completion(
            model=MODEL, messages=messages, tools=TOOLS,
            tool_choice="auto",
        )
        return follow_up.choices[0].message.content or "The tool operation finished."
    except APIStatusError as exc:
        if exc.status_code in (429, 500, 502, 503, 504):
            return "The inventory action ran, but Gemini is temporarily unavailable (HTTP " + str(exc.status_code) + "). Please ask me to check the result before repeating the action."
        return "Gemini follow-up failed (HTTP " + str(exc.status_code) + "). Please check the inventory before repeating the action."


def refresh_all():
    items = inventory_records()
    cats = categories_list()
    return (
        dashboard_summary(),
        gr.update(value=inventory_rows(), headers=inventory_headers()),
        gr.update(value=inventory_rows(low_only=True), headers=inventory_headers()),
        [[c["id"], c["name"]] for c in cats],
        audit_rows(),
        gr.update(choices=[c["name"] for c in cats]),
        gr.update(choices=[r["name"] for r in items]),
        gr.update(choices=[c["name"] for c in cats]),
        gr.update(choices=[f["name"] for f in custom_fields_list()]),
    )


def add_item_ui(name, category, quantity, threshold, unit, expiry_date):
    try:
        message = save_item(name, category, quantity, threshold, unit, expiry_date)
        return message, dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()
    except Exception as exc:
        return f"Could not save item: {exc}", dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()


def adjust_ui(name, category, delta):
    try:
        message = adjust_item(name, category, delta)
        return message, dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()
    except Exception as exc:
        return f"Could not adjust stock: {exc}", dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()


def custom_field_ui(name, field_type):
    try:
        message = add_custom_field(name, field_type)
        return message, gr.update(headers=inventory_headers(), value=inventory_rows()), gr.update(headers=inventory_headers(), value=inventory_rows(True)), gr.update(choices=[f["name"] for f in custom_fields_list()])
    except Exception as exc:
        return f"Could not add column: {exc}", gr.update(), gr.update(), gr.update()


def move_field_ui(name, direction):
    try:
        message = move_custom_field(name, direction)
        return message, gr.update(headers=inventory_headers(), value=inventory_rows()), gr.update(headers=inventory_headers(), value=inventory_rows(True))
    except Exception as exc:
        return f"Could not move column: {exc}", gr.update(), gr.update()


def remove_field_ui(name):
    try:
        message = remove_custom_field(name)
        return message, gr.update(headers=inventory_headers(), value=inventory_rows()), gr.update(headers=inventory_headers(), value=inventory_rows(True)), gr.update(choices=[f["name"] for f in custom_fields_list()])
    except Exception as exc:
        return f"Could not remove column: {exc}", gr.update(), gr.update(), gr.update()


def save_table_ui(data):
    try:
        message = save_inventory_table(data)
        return message, gr.update(value=inventory_rows()), gr.update(value=inventory_rows(True)), dashboard_summary(), audit_rows()
    except Exception as exc:
        return f"Could not save table: {exc}", gr.update(), gr.update(), dashboard_summary(), audit_rows()


def add_category_ui(name):
    try:
        message = add_category(name)
        cats = categories_list()
        return message, [[c["id"], c["name"]] for c in cats], gr.update(choices=[c["name"] for c in cats])
    except Exception as exc:
        return f"Could not add category: {exc}", [[c["id"], c["name"]] for c in categories_list()], gr.update()


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
.gradio-container {max-width: 1400px !important; margin: auto !important; padding: 18px !important;}
.tla-header {margin-bottom: 20px !important;}
.gradio-container .tab-nav button {font-weight: 650 !important; padding: 12px 18px !important;}
.gradio-container .block {border-radius: 14px !important;}
.gradio-container input, .gradio-container textarea {border-radius: 10px !important;}
.gradio-container button {border-radius: 10px !important; font-weight: 650 !important;}
/* Keep inventory columns readable on phones instead of squeezing every column. */
.gradio-container .table-wrap, .gradio-container .table-container, .gradio-container .dataframe-container {display:block !important; max-width:100% !important; overflow-x:auto !important; -webkit-overflow-scrolling:touch !important;}
.gradio-container .table-wrap table, .gradio-container .table-container table, .gradio-container .dataframe-container table {min-width:980px !important; width:max-content !important; table-layout:auto !important;}
.gradio-container .table-wrap th, .gradio-container .table-wrap td, .gradio-container .table-container th, .gradio-container .table-container td, .gradio-container .dataframe-container th, .gradio-container .dataframe-container td {white-space:nowrap !important; min-width:88px !important; padding:10px 12px !important;}
@media(max-width:600px) {.gradio-container {padding:8px !important;} .gradio-container .block {padding:10px !important;}}
"""


with gr.Blocks(title="The Liquid Affair | Inventory", css=APP_CSS) as demo:
    theme_styles = gr.HTML(theme_css("Neon Lounge"))
    gr.HTML(brand_header())
    with gr.Row():
        theme_selector = gr.Dropdown(
            choices=["Neon Lounge", "Tropical Luxe", "Classic Gold"],
            value="Neon Lounge", label="Choose your theme", scale=1
        )
    theme_selector.change(fn=theme_css, inputs=theme_selector, outputs=theme_styles)
    gr.Markdown("### Your bar, your stock — all in one place")
    gr.Markdown("Add products, track expiry dates, customize columns, and keep stock levels up to date.")
    with gr.Tabs():
        with gr.Tab("Dashboard"):
            summary = gr.Markdown()
            refresh_button = gr.Button("Refresh dashboard")
            gr.Markdown("#### Low-stock items")
            low_table = gr.Dataframe(
                headers=["ID", "Item", "Category", "Quantity", "Unit", "Reorder at", "Expiry date", "Updated"],
                interactive=False, wrap=False
            )
        with gr.Tab("Inventory"):
            gr.Markdown("#### Add item or set its exact stock")
            with gr.Row():
                item_name = gr.Textbox(label="Item name", placeholder="Mango Juice")
                item_category = gr.Dropdown(label="Category", choices=[], allow_custom_value=True, info="Choose an existing category or type a new one.")
            with gr.Row():
                item_quantity = gr.Number(label="Exact quantity", value=0, precision=0)
                item_unit = gr.Textbox(label="Unit", value="bottles")
                item_threshold = gr.Number(label="Reorder threshold", value=5, precision=0)
            item_expiry = gr.Textbox(label="Expiry date (YYYY-MM-DD, optional)", placeholder="2027-06-30")
            save_button = gr.Button("Save item", variant="primary")
            save_status = gr.Markdown()
            gr.Markdown("#### Adjust existing stock (+/-)")
            with gr.Row():
                adjust_name = gr.Dropdown(label="Select existing item", choices=[], interactive=True)
                adjust_category = gr.Dropdown(label="Select category", choices=[], interactive=True)
                adjust_delta = gr.Number(label="Stock change (+/-)", precision=0)
            adjust_button = gr.Button("Adjust stock")
            adjust_status = gr.Markdown()
            gr.Markdown("#### All inventory")
            gr.Markdown("Edit quantities, expiry dates, and custom values directly in the table, then save.")
            inventory_table = gr.Dataframe(
                headers=inventory_headers(), interactive=True, wrap=False
            )
            save_table_button = gr.Button("Save table changes", variant="primary")
            table_status = gr.Markdown()
            gr.Markdown("#### Customize your columns")
            gr.Markdown("Add fields such as Brand, Supplier, Bottle size, Storage location, Batch number, or Notes. Move or remove custom columns whenever you need.")
            with gr.Row():
                custom_field_name = gr.Textbox(label="New column name", placeholder="Supplier")
                custom_field_type = gr.Dropdown(label="Value type", choices=["text", "number", "date"], value="text")
                add_field_button = gr.Button("Add column", variant="primary")
            field_status = gr.Markdown()
            with gr.Row():
                manage_field = gr.Dropdown(label="Custom column", choices=[f["name"] for f in custom_fields_list()])
                move_direction = gr.Radio(choices=["up", "down"], value="up", label="Move")
                move_field_button = gr.Button("Move column")
                remove_field_button = gr.Button("Remove column", variant="stop")
        with gr.Tab("Categories"):
            category_name = gr.Textbox(label="New category", placeholder="Juices, Syrups, Glassware...")
            category_button = gr.Button("Add category", variant="primary")
            category_status = gr.Markdown()
            category_table = gr.Dataframe(headers=["ID", "Category"], interactive=False)
        with gr.Tab("AI assistant"):
            gr.ChatInterface(
                fn=chat_fn,
                title="Ask The Liquid Affair",
                description="Try: 'Add 24 bottles of Mango Juice under Juices' or 'Show low stock'.",
                examples=["Show low stock", "List inventory", "Add category Syrups",
                          "Reduce Straws by 5 in Straws"],
            )
        with gr.Tab("Stock history"):
            audit_table = gr.Dataframe(
                headers=["Date", "Item", "Action", "Old quantity", "New quantity", "Note"],
                interactive=False, wrap=True
            )

    refresh_outputs = [summary, inventory_table, low_table, category_table, audit_table, item_category, adjust_name, adjust_category, manage_field]
    demo.load(refresh_all, outputs=refresh_outputs)
    refresh_button.click(refresh_all, outputs=refresh_outputs)
    save_button.click(
        add_item_ui,
        inputs=[item_name, item_category, item_quantity, item_threshold, item_unit, item_expiry],
        outputs=[save_status, summary, inventory_table, low_table, audit_table],
    )
    adjust_button.click(
        adjust_ui, inputs=[adjust_name, adjust_category, adjust_delta],
        outputs=[adjust_status, summary, inventory_table, low_table, audit_table],
    ).then(refresh_all, outputs=refresh_outputs)
    save_table_button.click(
        save_table_ui, inputs=[inventory_table],
        outputs=[table_status, inventory_table, low_table, summary, audit_table],
    )
    add_field_button.click(
        custom_field_ui, inputs=[custom_field_name, custom_field_type],
        outputs=[field_status, inventory_table, low_table, manage_field],
    )
    move_field_button.click(
        move_field_ui, inputs=[manage_field, move_direction],
        outputs=[field_status, inventory_table, low_table],
    )
    remove_field_button.click(
        remove_field_ui, inputs=[manage_field],
        outputs=[field_status, inventory_table, low_table, manage_field],
    )
    category_button.click(
        add_category_ui, inputs=[category_name],
        outputs=[category_status, category_table, item_category],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        auth=(APP_USERNAME, APP_PASSWORD),
    )
