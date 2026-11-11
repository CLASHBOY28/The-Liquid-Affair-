-- Add a named user to each shared inventory history event.
alter table inventory_audit
  add column if not exists actor text;

update inventory_audit
set actor = 'Legacy'
where actor is null;

alter table inventory_audit
  alter column actor set default 'Unknown',
  alter column actor set not null;

