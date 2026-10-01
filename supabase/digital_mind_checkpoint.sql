-- DIGITAL_MIND current atomic checkpoint store.
-- Applied to project wwbnolaiegrkojrdjbbu on 2026-10-01.
-- Service-only: anon/authenticated have no table privileges.

create table if not exists public.digital_mind_checkpoint_current (
  id smallint primary key default 1 check (id = 1),
  generation bigint not null default 0,
  checkpoint_id text not null unique,
  package_version text not null,
  git_commit text,
  source_sha256 text not null check (source_sha256 ~ '^[0-9a-f]{64}$'),
  state_sha256 text not null check (state_sha256 ~ '^[0-9a-f]{64}$'),
  payload_sha256 text not null check (payload_sha256 ~ '^[0-9a-f]{64}$'),
  payload jsonb not null,
  byte_size bigint not null check (byte_size > 0),
  materialized_at timestamptz not null default now(),
  verified_at timestamptz,
  verification_error text,
  check (payload ->> 'format' = 'digital-mind-atomic-checkpoint-v1'),
  check (payload -> 'integrity' ->> 'checkpoint_id' = checkpoint_id),
  check (payload -> 'integrity' ->> 'payload_sha256' = payload_sha256),
  check (payload -> 'identity' ->> 'package_version' = package_version),
  check (payload -> 'identity' ->> 'source_sha256' = source_sha256),
  check (payload -> 'identity' ->> 'state_sha256' = state_sha256),
  check (coalesce(payload -> 'identity' ->> 'git_commit', '') = coalesce(git_commit, ''))
);

alter table public.digital_mind_checkpoint_current enable row level security;

revoke all on table public.digital_mind_checkpoint_current from anon, authenticated;
revoke delete, references, trigger, truncate
  on table public.digital_mind_checkpoint_current from service_role;
grant select, insert, update
  on table public.digital_mind_checkpoint_current to service_role;

drop policy if exists digital_mind_checkpoint_service_only
  on public.digital_mind_checkpoint_current;

create policy digital_mind_checkpoint_service_only
on public.digital_mind_checkpoint_current
for all
to service_role
using (true)
with check (true);

create or replace function public.digital_mind_checkpoint_prepare()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
begin
  new.id := 1;
  new.generation := case when tg_op = 'UPDATE' then old.generation + 1 else 1 end;
  new.materialized_at := now();
  if new.verification_error is null then
    new.verified_at := coalesce(new.verified_at, now());
  else
    new.verified_at := null;
  end if;
  return new;
end;
$$;

revoke all on function public.digital_mind_checkpoint_prepare()
from public, anon, authenticated;
grant execute on function public.digital_mind_checkpoint_prepare()
to service_role;

drop trigger if exists digital_mind_checkpoint_prepare_trg
on public.digital_mind_checkpoint_current;

create trigger digital_mind_checkpoint_prepare_trg
before insert or update on public.digital_mind_checkpoint_current
for each row execute function public.digital_mind_checkpoint_prepare();

create or replace function public.digital_mind_checkpoint_sync_runtime()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
begin
  update public.yado_rc7_runtime_state
  set boundaries =
      jsonb_set(
        jsonb_set(
          coalesce(boundaries, '{}'::jsonb),
          '{checkpoint_materialized}',
          'true'::jsonb,
          true
        ),
        '{digital_mind_checkpoint}',
        jsonb_build_object(
          'format', new.payload ->> 'format',
          'checkpoint_id', new.checkpoint_id,
          'generation', new.generation,
          'package_version', new.package_version,
          'git_commit', new.git_commit,
          'source_sha256', new.source_sha256,
          'state_sha256', new.state_sha256,
          'payload_sha256', new.payload_sha256,
          'byte_size', new.byte_size,
          'materialized_at', new.materialized_at,
          'verified_at', new.verified_at,
          'verification_error', new.verification_error
        ),
        true
      ),
      updated_at = now()
  where id = 1;
  return new;
end;
$$;

revoke all on function public.digital_mind_checkpoint_sync_runtime()
from public, anon, authenticated;
grant execute on function public.digital_mind_checkpoint_sync_runtime()
to service_role;

drop trigger if exists digital_mind_checkpoint_sync_runtime_trg
on public.digital_mind_checkpoint_current;

create trigger digital_mind_checkpoint_sync_runtime_trg
after insert or update on public.digital_mind_checkpoint_current
for each row execute function public.digital_mind_checkpoint_sync_runtime();

comment on table public.digital_mind_checkpoint_current is
'Service-only singleton storing the complete current DIGITAL_MIND atomic checkpoint. No anon/authenticated access.';
