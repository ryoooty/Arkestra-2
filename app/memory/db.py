"""
db.py — доступ к БД/миграции.
Таблицы (минимально):
- messages(id, user_id, role, text, ts, approved)
- facts(id, user_id, key, value, ts)
- notes(id, user_id, text, tags, ts)
- aliases(id, user_id, alias, is_primary, short_desc, ts)
- feedback(id, msg_id, kind up|down|text, text?, ts)
- neuro_logs(id, session_id, levels_json, preset_json, ts)
- bandit_stats(intent, kind, wins, plays, updated_at)
- tools(name PK, title, description, instruction, entrypoint, enabled)
- env_sessions(id, channel, chat_id, participants_json, started_at, last_seen)
- env_facts(id, env_id, key, value, importance, updated_at)
- sleep_batches(id, started_at, finished_at, from_seen_at, to_seen_at, processed_count, status)
"""
