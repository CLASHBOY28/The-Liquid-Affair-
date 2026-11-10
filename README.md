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

A private inventory management app for The Liquid Affair, built with Gradio, OpenAI, and Supabase.

## Features

- Inventory dashboard with item counts and low-stock alerts
- Add items and set exact quantities
- Increase or decrease stock with zero-stock protection
- Custom categories
- Stock audit history
- AI assistant for natural-language inventory operations
- Login protection using Gradio authentication

## Files

- \`app.py\` — Dashboard, database operations, and AI assistant
- \`schema.sql\` — Supabase database tables and indexes
- \`requirements.txt\` — Python dependencies
- \`.env.example\` — Names of required environment variables (no secrets)
- \`index.html\` — Earlier standalone browser demo; it is not connected to Supabase

## 1. Set up Supabase

1. Create a Supabase project.
2. Open **SQL Editor** in the Supabase dashboard.
3. Paste and run the complete contents of \`schema.sql\`.
4. In **Project Settings → API**, copy the Project URL and service-role key. Keep the service-role key private.

## 2. Configure secrets

For local development, copy \`.env.example\` to \`.env\` and fill in the values. Do not commit \`.env\`.

Required variables:

- \`SUPABASE_URL\`
- \`SUPABASE_SERVICE_ROLE_KEY\`
- \`OPENAI_API_KEY\`
- \`APP_USERNAME\`
- \`APP_PASSWORD\`

Optional:

- \`OPENAI_MODEL\` — defaults to \`gpt-4o-mini\`
- \`PORT\` — defaults to \`7860\`

Use a long, unique password for \`APP_PASSWORD\`.

## 3. Run locally

Requires Python 3.10 or newer.

\`\`\`bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
\`\`\`

Open the local URL printed by Gradio and sign in with \`APP_USERNAME\` and \`APP_PASSWORD\`.

## 4. Deploy on Hugging Face Spaces

This repository includes the Gradio Space metadata at the top of this README.

1. Create a **private Gradio Space** on Hugging Face.
2. Upload/push this repository to that Space (or connect the repository if that option is available).
3. In the Space, open **Settings → Variables and secrets**.
4. Add the five required secrets listed above. Add \`OPENAI_MODEL\` only if you want a model other than the default.
5. Restart/rebuild the Space and open its app URL.

The GitHub repository is prepared for this deployment, but the Space itself and its secrets must be created in your hosting account. Do not put credentials in GitHub files or chat messages.

## Security notes

- The Supabase service-role key bypasses Row Level Security. It is used only by the server-side Python process.
- Keep the Space private and retain the app login. Do not publish the service-role key, OpenAI key, or \`.env\`.
- No anonymous Supabase read/write policies are created.
- Audit entries are written for item creation, exact quantity changes, and stock adjustments.

## Scope

This release covers inventory and stock operations. Event bookings, quotations, event costing, expenses, staff scheduling, and sales reports are not implemented yet and should be added as separate modules after the inventory app is deployed and tested.
