import json
import os
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client

load_dotenv()  # Loads a local .env during development; hosting secrets take precedence.

# Configure these as server-side environment variables. Never expose the
# Supabase service-role key or OpenAI key in browser JavaScript.
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOOLS = [
    {"type": "function", "function": {"name": "add_or_update_item",
     "description": "Add an inventory item or set its exact quantity. Creates its category if needed.",
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
     "description": "List all inventory items and their categories.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "low_stock",
     "description": "List items at or below their reorder threshold.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "add_category",
     "description": "Create an inventory category.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                    "required": ["name"]}}},
    {"type": "function", "function": {"name": "list_categories",
     "description": "List all inventory categories.",
     "parameters": {"type": "object", "properties": {}}}},
]


def get_or_create_category(category_name: str) -> int:
    name = category_name.strip().lower()
    if not name:
        raise ValueError("Category name cannot be empty.")
    result = supabase.table("categories").select("id").eq("name", name).limit(1).execute()
    if result.data:
        return result.data[0]["id"]
    inserted = supabase.table("categories").insert({"name": name}).execute()
    if not inserted.data:
        raise RuntimeError("Category could not be created.")
    return inserted.data[0]["id"]


def find_item(name: str, category: str):
    # A lookup must not create a category as a side effect.
    category_name = category.strip().lower()
    cat = (supabase.table("categories").select("id")
           .eq("name", category_name).limit(1).execute())
    if not cat.data:
        return None
    result = (supabase.table("inventory_items")
              .select("id, quantity, name, category_id")
              .eq("name", name.strip())
              .eq("category_id", cat.data[0]["id"]).limit(1).execute())
    return result.data[0] if result.data else None


def write_audit(item_name, action, old_qty=None, new_qty=None, note=None):
    # Audit failures are surfaced in server logs rather than hiding them.
    supabase.table("inventory_audit").insert({
        "item_name": item_name, "action": action,
        "old_quantity": old_qty, "new_quantity": new_qty, "note": note,
    }).execute()


def execute_tool(name: str, args: dict[str, Any]) -> str:
    try:
        if name == "add_or_update_item":
            item_name = str(args["name"]).strip()
            category = str(args["category"]).strip()
            quantity = int(args["quantity"])
            threshold = int(args.get("reorder_threshold", 5))
            unit = str(args.get("unit", "pcs")).strip() or "pcs"
            if not item_name or not category or quantity < 0 or threshold < 0:
                return "ERROR: Enter a name, category, and non-negative quantities."
            cat_id = get_or_create_category(category)
            existing = find_item(item_name, category)
            if existing:
                old_qty = existing["quantity"]
                (supabase.table("inventory_items").update({
                    "quantity": quantity, "reorder_threshold": threshold, "unit": unit
                }).eq("id", existing["id"]).execute())
                write_audit(item_name, "set_quantity", old_qty, quantity, f"category={category}")
                return f"Updated {item_name}: {old_qty} -> {quantity} {unit}."
            supabase.table("inventory_items").insert({
                "name": item_name, "category_id": cat_id, "quantity": quantity,
                "reorder_threshold": threshold, "unit": unit,
            }).execute()
            write_audit(item_name, "create", None, quantity, f"category={category}")
            return f"Added {item_name} to {category}: {quantity} {unit}."

        if name == "adjust_stock":
            item_name = str(args["name"]).strip()
            category = str(args["category"]).strip()
            delta = int(args["delta"])
            existing = find_item(item_name, category)
            if not existing:
                return f"{item_name} was not found in {category}. Add it first."
            old_qty = existing["quantity"]
            new_qty = old_qty + delta
            if new_qty < 0:
                return f"Cannot reduce {item_name} below zero. Current stock: {old_qty}."
            (supabase.table("inventory_items").update({"quantity": new_qty})
             .eq("id", existing["id"]).execute())
            write_audit(item_name, "adjust", old_qty, new_qty, f"delta={delta}")
            return f"{item_name}: {old_qty} -> {new_qty}."

        if name == "list_inventory":
            result = (supabase.table("inventory_items")
                      .select("name, quantity, unit, reorder_threshold, categories(name)")
                      .order("name").execute())
            return json.dumps(result.data, default=str)

        if name == "low_stock":
            result = (supabase.table("inventory_items")
                      .select("name, quantity, unit, reorder_threshold, categories(name)")
                      .execute())
            low = [row for row in result.data
                   if row["quantity"] <= row["reorder_threshold"]]
            return json.dumps(low, default=str)

        if name == "add_category":
            category_id = get_or_create_category(str(args["name"]))
            return f"Category created or already exists (id={category_id})."

        if name == "list_categories":
            result = supabase.table("categories").select("id, name").order("name").execute()
            return json.dumps(result.data, default=str)

        return f"Unknown tool: {name}"
    except Exception as exc:
        print(f"Inventory tool error ({name}): {type(exc).__name__}: {exc}")
        return f"ERROR: The inventory operation failed ({type(exc).__name__}). Check server logs."


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
You can add or set stock, adjust stock, list inventory, show low stock, and manage categories.
Always use the provided tools for inventory data and changes. Never invent stock values.
A request to add stock means add_or_update_item sets the exact quantity; a request to
increase/decrease stock means adjust_stock. Ask one short question if an item or category
is ambiguous. Be concise and clearly report the result returned by the tool.
"""


def chat_fn(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": message})
    response = openai_client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS,
        tool_choice="auto", parallel_tool_calls=False, temperature=0.1,
    )
    assistant_message = response.choices[0].message
    if not assistant_message.tool_calls:
        return assistant_message.content or "I couldn't produce a response."

    messages.append(assistant_message)
    for call in assistant_message.tool_calls:
        try:
            arguments = json.loads(call.function.arguments or "{}")
            result = execute_tool(call.function.name, arguments)
        except (ValueError, TypeError, json.JSONDecodeError):
            result = "ERROR: The requested operation had invalid arguments."
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    follow_up = openai_client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS,
        tool_choice="auto", temperature=0.1,
    )
    return follow_up.choices[0].message.content or "The operation finished."


demo = gr.ChatInterface(
    fn=chat_fn,
    title="The Liquid Affair | Inventory Assistant",
    description="Manage bar stock, categories, and low-stock alerts using natural language.",
    examples=[
        "Add 24 bottles of Mango Juice under Juices",
        "Reduce Straws by 5 in the Straws category",
        "Show low stock",
        "Add category Syrups",
        "List inventory",
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
