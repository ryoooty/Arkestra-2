# Arkestra — локальная ИИ-персона

Arkestra — это локальная мультиагентная связка, где junior-модель управляет процессом и нейропрофилем, а senior-модель формирует итоговый ответ и вызывает инструменты. Проект включает RAG, долговременную память и мягкую защиту контента, чтобы его можно было запускать полностью оффлайн.

## Стек

- **Senior**: Mistral-7B — финальный исполнитель, стиль, `tool_calls`
- **Junior**: DeepSeek-R1-1.5B / ≤3B — диспетчер, `style_directive`, `neuro_update`, `tools_hint`
- **RAG**: FAISS хранилище + rerankер e5-small (short / temp / long контуры)
- **Память**: SQLite (messages, notes, facts, aliases, feedback, bandit, env, sleep_batches, reminders)
- **Нейропрофиль**: 10 медиаторов (dopamine … histamine) с min/base/max и bias-map
- **Guard**: мягкая цензура (PII, profanity → `***`)
- **Админ / API**: FastAPI (`/chat`, `/tools`, `/ui/tools`, `/metrics`, `/health`)
- **Планировщик**: APScheduler (напоминания + авто-сон)

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # при необходимости: sentence-transformers, faiss-cpu
python -m scripts.migrate
python -m scripts.seed_rag        # опционально: первичный прогрев RAG
```

> Если требуется отправка через Telegram, создайте `config/tg.yaml` с токеном и `self_chat_id`.

## Запуск

### CLI (интерактивный REPL)

```bash
make run
```

Горячие клавиши: `/up`, `/down`, `/fb`, `/ok`, `/sleep`. Планировщик напоминаний и авто-сна стартует автоматически.

### REST API и админка инструментов

```bash
make api
```

- UI: <http://localhost:8000/ui/tools>
- API: `POST /chat`, `GET/POST /tools`, `GET /metrics`, `GET /health`

## Основной поток обработки

1. `orchestrator.handle_user` формирует `env_brief` (канал, участники, факты окружения).
2. **Junior** (`junior.generate`) выдаёт `intent`, `tools_hint`, `tools_request`, `style_directive`, `neuro_update`, `rag_query`.
3. `neuro.set_levels` применяет уровни медиаторов и пересчитывает стили.
4. RAG ищет по FAISS и ранжирует через e5-small.
5. **Senior** (`senior.generate_structured`) собирает финальный ответ, `tool_calls`, `memory`, `plan`.
6. `tools_runner.run_all` выполняет запрошенные инструменты и при необходимости вызывает `senior.refine_with_results`.
7. `guard.soft_censor` маскирует чувствительные данные перед выводом.
8. Асинхронно обновляются сообщения, bandit/feedback и нейроданные.

## Авто-сон и консолидация памяти

Скрипт `scripts.consolidate_sleep.run_sleep_batch` переводит заметки short→temp (формирует `day_summaries` и curated facts), обновляет TTL temp→long, освежает RAG, сбрасывает нейропрофиль и экспортирует датасеты для обучения.

### Планировщик сна через APScheduler

```bash
python scripts/scheduler.py
```

Задача автоматически запускает сон ежедневно в 04:00 по серверному времени. Нажмите `Ctrl+C`, чтобы остановить фоновый планировщик.

#### Резервный cron

```cron
0 4 * * * cd /path/to/Arkestra-2 && /usr/bin/env python scripts/consolidate_sleep.py
```

## Инструменты

Из коробки доступны:

- `note.create {text, tags?}` — сохраняет заметку в БД
- `reminder.create {when, title, channel?}` — планирует напоминание (persist через APScheduler)
- `messages.search_by_date {date, span_days?}` — выборка истории
- `alias.add {alias, short_desc?}` / `alias.set_primary {alias}`
- `tg.message.send {to, text}` — отправка в Telegram (опционально)

Добавление/редактирование инструментов — через UI `/ui/tools` или API `/tools`. Если junior запросил отсутствующий инструмент (`tools_request`), senior попросит добавить его и объяснит, зачем он нужен.

## Конфигурация

- `config/llm.yaml` — модели и параметры генерации
- `config/persona.yaml` — медиаторы, bias-map, sleep-reset
- `config/router.yaml` — настройки RAG (маршрутизация, `top_k`, `rerank_top_k`)
- `config/guard.yaml` — паттерны масок
- `config/tg.yaml` — Telegram-конфигурация (опционально)

## Тесты и метрики

```bash
make test  # запускает pytest: guard, neuro, bandit, rag, server, e2e с моками LLM
```

Prometheus-метрики доступны по `GET /metrics`, состояние системы — `GET /health`.

## Экспорт датасетов обучения

- Senior (SFT): `python -m scripts.export_sft` → `data/sft_staging.jsonl`
- Junior (micro-LoRA): `python -m scripts.export_junior_lora` → `data/junior_lora.jsonl`

Оба скрипта вызываются из режима сна и могут запускаться вручную.

## FAQ

### Зачем junior, если инструменты доступны senior-модели?

Junior выполняет роль диспетчера: решает, какой инструмент нужен, задаёт стиль ответа (`style_directive`) и обновляет нейропрофиль. Это повышает осознанность и экономит токены senior.

### Как работает осознанность окружения?

Через `env_sessions` и `env_facts`. В оба промпта попадает краткий `env_brief`: где находимся, кто рядом, какие активные темы.

### Как включить авто-сон вручную?

В CLI доступна команда `/sleep`. Планировщик в CLI и API стартует автоматически.

### Пример системного промпта для senior (сокращённо)

```
You are **Arkestra**, a local evolving AI persona (you know you are an AI).
Your role: compose the final user-facing reply, decide `tool_calls`, record memory, suggest a short plan.

Constraints and style:
- Respect the junior’s `style_directive` and the neuro `preset` (temperature/length/structure/empathy/humor, etc.).
- Use tools exactly per TOOLS_INSTRUCTIONS when needed; do not invent tools.
- Keep replies human, warm, precise; if tasky → be concise and structured.
- Apply alias policy (if present); avoid PII leakage and profanity (softly masked later).
- If JUNIOR.tools_request names missing tools, politely ask the user to add them and explain why.

Context:
ENV: {channel, participants, brief env_facts}
PRESET: {temperature, max_tokens, biases}
STYLE: "{style_directive}"
RAG_HITS: [{id,text,meta}...]
HISTORY: [{role,text}...]
JUNIOR_JSON: {...}

Output JSON only:
{
  "text": "...",
  "tool_calls": [{"name":"tool.name","args":{...}}]?,
  "memory": [{"kind":"note|fact","data":{...}}]?,
  "plan": ["..."]?
}
```
