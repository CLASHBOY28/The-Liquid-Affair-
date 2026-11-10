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

if not APP_USERNAME or not APP_PASSWORD:
    raise RuntimeError("Set APP_USERNAME and APP_PASSWORD before starting the app.")

ai_client = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
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
                return ai_client.chat.completions.create(**request)
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code == 429:
                    break
                if exc.status_code not in (500, 502, 503, 504) or attempt + 1 >= max_attempts:
                    break
                time.sleep(1)
    raise last_error


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


def inventory_records():
    result = (supabase.table("inventory_items")
              .select("id, name, quantity, unit, reorder_threshold, updated_at, categories(name)")
              .order("name").execute())
    return result.data or []


def inventory_rows(low_only: bool = False):
    records = inventory_records()
    if low_only:
        records = [r for r in records if r["quantity"] <= r["reorder_threshold"]]
    rows = []
    for item in records:
        category = item.get("categories") or {}
        rows.append([
            item["id"], item["name"], category.get("name", "Uncategorised"),
            item["quantity"], item.get("unit") or "pcs",
            item["reorder_threshold"], item.get("updated_at", "")
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


def save_item(name, category, quantity, threshold, unit):
    name = (name or "").strip()
    category = (category or "").strip()
    unit = (unit or "pcs").strip() or "pcs"
    if not name or not category:
        raise ValueError("Enter an item name and choose a category.")
    if quantity is None or int(quantity) < 0 or threshold is None or int(threshold) < 0:
        raise ValueError("Quantity and reorder threshold must be zero or greater.")
    quantity, threshold = int(quantity), int(threshold)
    cat_id = category_id_for(category, create=True)
    found = (supabase.table("inventory_items").select("id, quantity")
             .eq("name", name).eq("category_id", cat_id).limit(1).execute())
    if found.data:
        old = found.data[0]["quantity"]
        (supabase.table("inventory_items").update({
            "quantity": quantity, "reorder_threshold": threshold,
            "unit": unit, "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", found.data[0]["id"]).execute())
        write_audit(name, "set_quantity", old, quantity, "Dashboard")
        return f"Updated {name}: {old} → {quantity} {unit}."
    (supabase.table("inventory_items").insert({
        "name": name, "category_id": cat_id, "quantity": quantity,
        "reorder_threshold": threshold, "unit": unit
    }).execute())
    write_audit(name, "create", None, quantity, "Dashboard")
    return f"Added {name}: {quantity} {unit}."


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
         "unit": {"type": "string"}},
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
                             args.get("reorder_threshold", 5), args.get("unit", "pcs"))
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
    return (
        dashboard_summary(),
        inventory_rows(),
        inventory_rows(low_only=True),
        [[c["id"], c["name"]] for c in categories_list()],
        audit_rows(),
        gr.update(choices=[c["name"] for c in categories_list()]),
    )


def add_item_ui(name, category, quantity, threshold, unit):
    try:
        message = save_item(name, category, quantity, threshold, unit)
        return message, dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()
    except Exception as exc:
        return f"Could not save item: {exc}", dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()


def adjust_ui(name, category, delta):
    try:
        message = adjust_item(name, category, delta)
        return message, dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()
    except Exception as exc:
        return f"Could not adjust stock: {exc}", dashboard_summary(), inventory_rows(), inventory_rows(True), audit_rows()


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


with gr.Blocks(title="The Liquid Affair | Inventory", css="") as demo:
    theme_styles = gr.HTML(theme_css("Neon Lounge"))
    gr.HTML(brand_header())
    with gr.Row():
        theme_selector = gr.Dropdown(
            choices=["Neon Lounge", "Tropical Luxe", "Classic Gold"],
            value="Neon Lounge", label="Choose your theme", scale=1
        )
    theme_selector.change(fn=theme_css, inputs=theme_selector, outputs=theme_styles)
    gr.Markdown("Manage stock, categories, reorder alerts, and stock history.")
    with gr.Tabs():
        with gr.Tab("Dashboard"):
            summary = gr.Markdown()
            refresh_button = gr.Button("Refresh dashboard")
            gr.Markdown("#### Low-stock items")
            low_table = gr.Dataframe(
                headers=["ID", "Item", "Category", "Quantity", "Unit", "Reorder at", "Updated"],
                interactive=False, wrap=True
            )
        with gr.Tab("Inventory"):
            gr.Markdown("#### Add item or set its exact stock")
            with gr.Row():
                item_name = gr.Textbox(label="Item name", placeholder="Mango Juice")
                item_category = gr.Dropdown(label="Category", choices=[], allow_custom_value=True)
            with gr.Row():
                item_quantity = gr.Number(label="Exact quantity", value=0, precision=0)
                item_unit = gr.Textbox(label="Unit", value="bottles")
                item_threshold = gr.Number(label="Reorder threshold", value=5, precision=0)
            save_button = gr.Button("Save item", variant="primary")
            save_status = gr.Markdown()
            gr.Markdown("#### Adjust existing stock (+/-)")
            with gr.Row():
                adjust_name = gr.Textbox(label="Item name")
                adjust_category = gr.Textbox(label="Exact category")
                adjust_delta = gr.Number(label="Change (e.g. -2 or 10)", precision=0)
            adjust_button = gr.Button("Adjust stock")
            adjust_status = gr.Markdown()
            gr.Markdown("#### All inventory")
            inventory_table = gr.Dataframe(
                headers=["ID", "Item", "Category", "Quantity", "Unit", "Reorder at", "Updated"],
                interactive=False, wrap=True
            )
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

    refresh_outputs = [summary, inventory_table, low_table, category_table, audit_table, item_category]
    demo.load(refresh_all, outputs=refresh_outputs)
    refresh_button.click(refresh_all, outputs=refresh_outputs)
    save_button.click(
        add_item_ui,
        inputs=[item_name, item_category, item_quantity, item_threshold, item_unit],
        outputs=[save_status, summary, inventory_table, low_table, audit_table],
    )
    adjust_button.click(
        adjust_ui, inputs=[adjust_name, adjust_category, adjust_delta],
        outputs=[adjust_status, summary, inventory_table, low_table, audit_table],
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
