-- Run once in Supabase SQL Editor before deploying the required Event Date UI.
-- Existing sessions are backfilled with their original creation date.
alter table public.event_stock_events
  add column if not exists event_date date;

update public.event_stock_events
set event_date = created_at::date
where event_date is null;

alter table public.event_stock_events
  alter column event_date set not null;
