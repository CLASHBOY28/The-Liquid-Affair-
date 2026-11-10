# The Liquid Affair

Inventory management assistant for The Liquid Affair, built with Gradio, OpenAI, and Supabase.

## Project files

- `app.py` — Gradio chat application and inventory tools.
- `schema.sql` — Supabase tables for categories, inventory, and audit history.
- `requirements.txt` — Python dependencies.
- `index.html` — Existing standalone page, preserved as-is.

## Setup

1. Create a Supabase project.
2. Open the Supabase SQL Editor and run the contents of `schema.sql`.
3. Install dependencies with `pip install -r requirements.txt`.
4. Set these environment variables in your deployment's secret/environment settings:

   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `OPENAI_API_KEY`
   - Optional: `OPENAI_MODEL` (defaults to `gpt-4o-mini`)

5. Start the app with `python app.py`.

The app listens on `0.0.0.0` and uses the hosting platform's `PORT` value, defaulting to `7860`.

## Security

Keep all API keys in server-side environment variables. Never paste the Supabase service-role key or OpenAI key into `index.html`, browser JavaScript, Git commits, or public messages. The service-role key bypasses Supabase Row Level Security, so restrict access to the deployed Gradio app with authentication or hosting-level access controls before making it public.

## Current scope

The assistant supports creating categories, adding or setting item quantities, adjusting stock, listing inventory, and identifying low-stock items. Event bookings, quotations, expenses, and reporting can be added in later stages.
