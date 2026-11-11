---
title: The Liquid Affair Inventory
emoji: 🍸
colorFrom: yellow
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# The Liquid Affair — Inventory & Bar Operations

A private, mobile-friendly inventory and bar-operations web app built with Gradio and Supabase. Its AI assistant supports general conversation and can read or update inventory through guarded server-side tools.

## Features

- Inventory dashboard and low-stock alerts
- Add items, set exact quantities, and adjust stock
- Categories and customizable inventory columns
- Stock-change audit history
- AI assistant with selectable Groq, Gemini, and Cloudflare Workers AI providers
- Chat-based inventory lookup and updates
- Two individual logins with shared inventory and named user history

## Supabase setup

1. Create a Supabase project.
2. Open **SQL Editor** in the Supabase dashboard.
3. Run the complete contents of `schema.sql`.
4. Copy the Project URL and the service-role key from **Project Settings → API**. Keep the service-role key private.

## Environment variables

Set these in Railway's service Variables (or in a local `.env` file):

**Required**
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `APP_USERNAME`
- `APP_PASSWORD`

**Optional second login**
- `APP_USERNAME_2`
- `APP_PASSWORD_2`

**At least one AI provider**
- `GROQ_API_KEY` — Groq; defaults to `openai/gpt-oss-120b`
- `GEMINI_API_KEY` — Google Gemini; defaults to `gemini-3.8-flash`

**Optional fallback**
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_MODEL` — defaults to `@cf/meta/llama-3.3-70b-instruct-fp8-fast`

Optional model overrides: `GROQ_MODEL`, `GEMINI_MODEL`, and `GEMINI_FALLBACK_MODEL`. The defaults are documented in `.env.example`.

Set both `APP_USERNAME_2` and `APP_PASSWORD_2` in Railway Variables to enable the second login. The app also accepts the older Railway names `APP_USERNAME2` and `APP_PASSWORD2`. Keep the usernames distinct and use a separate long, unique password for each account. A partially configured pair is ignored if another complete pair exists; at least one complete pair is always required. Never commit `.env`, passwords, API keys, or the Supabase service-role key.

For named audit rows in an existing Supabase database, run `migration_add_audit_actor.sql` once in the Supabase SQL Editor. The app keeps logging the username in the existing note field until that migration is applied, so stock changes remain attributable during setup. The Stock History page shows the user, action, item, previous/new quantities, and note in one shared log.

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio and sign in with either configured username and its matching password.

## Deploy

The repository's `main` branch is connected to the Railway service. Set the required Supabase and login variables, then add at least one AI provider key in the Railway service's production Variables. A new deployment is needed after changing environment variables so the app process reloads them.

## Security notes

- The Supabase service-role key bypasses Row Level Security and is used only by the server-side Python process.
- Keep this service private and retain the app login.
- No anonymous Supabase read/write policies are created.
- Inventory mutations are recorded in `inventory_audit`.

## Scope

The current release implements inventory, categories, custom inventory columns, low-stock tracking, stock history, and the AI assistant's inventory tools. Event bookings, quotations, costing, expenses, staff scheduling, and sales reporting are not implemented in this release.

