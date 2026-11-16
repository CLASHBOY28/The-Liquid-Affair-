-- Event Stock now reserves stock immediately and writes matching audit rows.
-- Run this once in Supabase SQL Editor after deploying the matching app.py.
alter table event_stock_events
  add column if not exists stock_deducted boolean not null default false;

create or replace function public.event_stock_totals(p_columns jsonb, p_rows jsonb)
returns table(item_name text, quantity integer)
language plpgsql immutable as $$
declare
  item_idx integer;
  qty_idx integer;
  r jsonb;
  nm text;
  q numeric;
begin
  select ordinality - 1 into item_idx
  from jsonb_array_elements_text(p_columns) with ordinality
  where lower(trim(value)) = 'item' limit 1;
  select ordinality - 1 into qty_idx
  from jsonb_array_elements_text(p_columns) with ordinality
  where lower(trim(value)) = 'quantity' limit 1;
  if item_idx is null or qty_idx is null then
    raise exception 'Event sheet must keep Item and Quantity columns.';
  end if;
  for r in select value from jsonb_array_elements(coalesce(p_rows, '[]'::jsonb))
  loop
    nm := nullif(trim(coalesce(r ->> item_idx, '')), '');
    if nm is null then
      if nullif(trim(coalesce(r ->> qty_idx, '')), '') is not null then
        raise exception 'Choose an inventory item for every quantity entered.';
      end if;
      continue;
    end if;
    begin
      q := coalesce(nullif(trim(r ->> qty_idx), ''), '0')::numeric;
    exception when others then
      raise exception 'Quantity for "%" must be a whole number.', nm;
    end;
    if q < 0 or q <> trunc(q) or q > 2147483647 then
      raise exception 'Quantity for "%" must be a non-negative whole number.', nm;
    end if;
    item_name := nm;
    quantity := q::integer;
    return next;
  end loop;
end $$;

create or replace function public.event_stock_apply(
  p_event_id bigint,
  p_rows jsonb default null,
  p_actor text default 'Unknown',
  p_delete boolean default false
)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare
  ev event_stock_events%rowtype;
  old_rec record;
  new_rec record;
  inv_id bigint;
  inv_name text;
  inv_qty integer;
  old_qty integer;
  new_qty integer;
  delta integer;
  act text;
  note_text text;
begin
  select * into ev from event_stock_events where id = p_event_id for update;
  if not found then raise exception 'Event not found.'; end if;

  if ev.stock_deducted then
    -- The saved event rows already represent stock deducted from Inventory.
    for old_rec in
      select lower(trim(item_name)) as k, min(item_name) as nm, sum(quantity)::integer as qty
      from public.event_stock_totals(ev.columns, ev.rows)
      group by lower(trim(item_name))
    loop
      old_qty := old_rec.qty;
      new_qty := 0;
      if not p_delete then
        select coalesce(sum(t.quantity),0)::integer into new_qty
        from public.event_stock_totals(ev.columns, coalesce(p_rows, ev.rows)) t
        where lower(trim(t.item_name)) = old_rec.k;
      end if;
      delta := old_qty - new_qty;
      if delta <> 0 then
        select id, name, quantity into inv_id, inv_name, inv_qty
        from inventory_items
        where lower(trim(name)) = old_rec.k
        for update;
        if not found then raise exception 'Inventory item "%" no longer exists.', old_rec.nm; end if;
        if (select count(*) from inventory_items where lower(trim(name)) = old_rec.k) <> 1 then
          raise exception 'Inventory item "%" is ambiguous across categories.', old_rec.nm;
        end if;
        if inv_qty + delta < 0 then
          raise exception 'Not enough stock for "%". Available: %; additional required: %.',
            inv_name, inv_qty, -delta;
        end if;
        update inventory_items set quantity = inv_qty + delta, updated_at = now() where id = inv_id;
        act := case when delta > 0 then 'event_stock_return' else 'event_stock_issue' end;
        note_text := case when p_delete then 'Returned from deleted event: ' else 'Event Stock adjustment: ' end || ev.name;
        insert into inventory_audit(actor,item_name,action,old_quantity,new_quantity,note)
        values(coalesce(nullif(p_actor,''),'Unknown'),inv_name,act,inv_qty,inv_qty+delta,note_text);
      end if;
    end loop;
  end if;

  -- Apply newly added rows (or all rows for a legacy session not yet deducted).
  if not p_delete then
    for new_rec in
      select lower(trim(item_name)) as k, min(item_name) as nm, sum(quantity)::integer as qty
      from public.event_stock_totals(ev.columns, coalesce(p_rows, ev.rows))
      group by lower(trim(item_name))
    loop
      if ev.stock_deducted then
        select coalesce(sum(t.quantity),0)::integer into old_qty
        from public.event_stock_totals(ev.columns, ev.rows) t
        where lower(trim(t.item_name)) = new_rec.k;
      else
        old_qty := 0;
      end if;
      delta := new_rec.qty - old_qty;
      if delta > 0 then
        select id, name, quantity into inv_id, inv_name, inv_qty
        from inventory_items
        where lower(trim(name)) = new_rec.k
        for update;
        if not found then raise exception 'Inventory item "%" was not found.', new_rec.nm; end if;
        if (select count(*) from inventory_items where lower(trim(name)) = new_rec.k) <> 1 then
          raise exception 'Inventory item "%" is ambiguous across categories.', new_rec.nm;
        end if;
        if inv_qty < delta then
          raise exception 'Not enough stock for "%". Available: %; requested: %.',
            inv_name, inv_qty, delta;
        end if;
        update inventory_items set quantity = inv_qty - delta, updated_at = now() where id = inv_id;
        insert into inventory_audit(actor,item_name,action,old_quantity,new_quantity,note)
        values(coalesce(nullif(p_actor,''),'Unknown'),inv_name,'event_stock_issue',
               inv_qty,inv_qty-delta,'Allocated to Event Stock: ' || ev.name);
      end if;
    end loop;
  end if;

  if p_delete then
    delete from event_stock_events where id = p_event_id;
    insert into inventory_audit(actor,item_name,action,note)
    values(coalesce(nullif(p_actor,''),'Unknown'),'Event: ' || ev.name,'event_delete',
           'Deleted Event Stock session');
    return 'Deleted ' || ev.name || '. Returned its deducted stock to main Inventory.';
  end if;

  update event_stock_events
  set rows = coalesce(p_rows, ev.rows), stock_deducted = true, updated_at = now()
  where id = p_event_id;
  return 'Saved Event Stock for ' || ev.name || '; main Inventory was adjusted immediately.';
end $$;

revoke all on function public.event_stock_apply(bigint,jsonb,text,boolean) from public, anon, authenticated;
grant execute on function public.event_stock_apply(bigint,jsonb,text,boolean) to service_role;
