-- One-shot DB fixup: run ONCE per database, BEFORE starting the API on the
-- upstream-1.14 merge. Idempotent, but see the guard below.
--
-- Why:
-- This fork shipped its own migrations 15 (notebook-scoped search), 16
-- (rag_trace) and 17 (chunk-level search). Upstream v1.9.0-v1.14.0 shipped a
-- DIFFERENT 15-23 (credential.config, episode_profile.max_tokens,
-- transformation.model_id, source_insight timestamps, speaker_profile record
-- links, relative podcast audio paths, legacy podcast field drop, Docling
-- toggles).
--
-- AsyncMigrationManager tracks progress by COUNT, not by file identity
-- (`for i in range(current_version, len(self.up_migrations))`). A database
-- sitting at version 17 because it ran THIS FORK's 15-17 would therefore skip
-- upstream's 15-17 entirely and silently start at 18 - leaving the schema
-- without credential.config et al. and breaking credentials, podcast profiles
-- and per-transformation model selection at runtime, with no migration error.
--
-- The fork's migrations have been renumbered to 24-26 (after upstream's 23).
-- This script walks the recorded version back to 14 so that, on next API
-- startup, the manager replays upstream 15-23 and then re-applies the fork's
-- 24-26. Re-applying the fork's three is safe: they are written with
-- `DEFINE ... IF NOT EXISTS` / `REMOVE FUNCTION IF EXISTS` + `DEFINE`, so they
-- converge to the same schema whether or not the objects already exist.
--
-- BACK UP surreal_data/ FIRST. Upstream's 20, 21 and 22 perform irreversible
-- DATA conversions (speaker_config name -> record link, absolute -> relative
-- audio paths, legacy podcast provider/model fields dropped).
--
-- Usage (adjust host/credentials):
--   docker compose exec -T surrealdb /surreal sql \
--     --conn ws://localhost:8000 --user root --pass root \
--     --ns open_notebook --db open_notebook --pretty \
--     < scripts/fixup-fork-migration-numbers.surql

-- Guard: only rewind a database that is exactly at the fork's version 17.
-- On any other version this deletes nothing, so an accidental second run on an
-- already-migrated database (version 26) is a no-op.
DELETE _sbl_migrations
WHERE version > 14
  AND (SELECT VALUE math::max(version) FROM ONLY _sbl_migrations GROUP ALL) = 17;

SELECT version, applied_at FROM _sbl_migrations ORDER BY version;
