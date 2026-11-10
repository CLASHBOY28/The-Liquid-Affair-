-- Run this once in Supabase SQL Editor for an existing Liquid Affair database.
alter table public.inventory_items
  add column if not exists expiry_date date;

create index if not exists inventory_items_expiry_date_idx
  on public.inventory_items(expiry_date)
  where expiry_date is not null;
