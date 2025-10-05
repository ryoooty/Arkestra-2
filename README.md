
# Arkestra — локальная ИИ-персона без интернета и без API

**Arkestra** — это полностью офлайн-личность: все модели, память, поиск знаний и инструменты работают **на вашем ПК**, без веб-запросов и без локальных веб-серверов.
Система из двух моделей (junior + senior) думает пошагово, хранит «нейропрофиль» (химию настроения), умеет извлекать свои знания через RAG, вызывать локальные инструменты (заметки, напоминания, поиск истории), и выдаёт человекоподобные ответы. Главный акцент проект на воссоздании человеческих процессов в "мозге" нейросети. Это добивается использованием нескольких нейросетей и общению между ними.

---

## Возможности (коротко)

* 💬 Двухуровневое мышление: Junior → директивы, Senior → финальный ответ
* 🧠 Нейропрофиль из 10 медиаторов(имитация человечесских эмоций) → живой стиль и «настроение»
* 📚 Локальный RAG: FAISS + e5-small (короткий/временный/долгий контуры)
* 🗂 Память в SQLite: сообщения, заметки, факты, алиасы, фидбэк, бандит-метрики
* 🛠 Инструменты: `note.create`, `reminder.create`, `messages.search_by_date`, `alias.add/set_primary`, `tg.message.send` (расширяются)
* 🛡 Guard: мягкая цензура PII/бранных слов
* ⏰ Планировщик (APScheduler): напоминания, авто-сон с дообучением

---

## Стек и модели

* **Senior**: `mistralai/Mistral-7B-Instruct-v0.3` (локально из папки, 4-бит через bitsandbytes)
* **Junior**: маленькая GGUF-модель через `llama.cpp` (напр. Gemma-3n-E4B-it); можно заменить на Qwen/DeepSeek 1–3B
* **RAG-эмбеддинги**: `intfloat/e5-small-v2` (по умолчанию; можно переключить на Qwen3-Embedding-0.6B)
* **Поиск**: FAISS (CPU), rerank — sentence-transformers (CPU/GPU)
* **БД**: SQLite

---

## Установка (Windows-friendly)

```bash
# 1) Виртуальное окружение
python -m venv .venv
.\.venv\Scripts\activate

# 2) Библиотеки (PyTorch под вашу CUDA, затем остальное)
# Пример для CUDA 12.x (подберите колёса под свою версию):
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) Остальные зависимости проекта
pip install -r requirements.txt
# при необходимости: pip install sentence-transformers faiss-cpu

# 4) Миграции БД и первичное наполнение RAG
python -m scripts.migrate
python -m scripts.seed_rag   # опционально, можно позже
```

> **CUDA/bitsandbytes:** если увидите «`The installed version of bitsandbytes was compiled without GPU support`», значит поставлен CPU-вариант. Установите сборку bnb, совместимую с вашей CUDA, или используйте FP16/8-бит без bnb.

---

## Конфиг моделей (пример)

`config/llm.yaml`:

```yaml
junior:
  provider: llama-cpp
  model_path: "C:\\Users\\kosha\\models\\gemma3n-e4b\\google_gemma-3n-E4B-it-Q4_K_M.gguf"
  n_ctx: 32768        # если модель поддерживает, иначе 2048–8192
  n_gpu_layers: 0     # 0 = CPU; можно >0 для частичной загрузки на GPU при CUDA-сборке
  temperature: 0.2
  max_new_tokens: 96

senior:
  provider: transformers
  model_id: "C:\\Users\\kosha\\models\\mistral-7b-instruct-v0_3"
  load_in_4bit: true
  torch_dtype: "float16"
  device_map: "auto"
  temperature: 0.7
  max_new_tokens: 512

embeddings:
  model: "intfloat/e5-small-v2"   # или "Qwen/Qwen3-Embedding-0.6B"
  device: "cpu"                   # можно "cuda" при желании
```

---

## Запуск

### CLI (интерактивный REPL)

```bash
python -m app.io.cli
```

Команды: `/up`, `/down`, `/fb <текст>`, `/ok`, `/sleep`.
Планировщик напоминаний/«сна» стартует автоматически.

---

## Основной поток выполнения

1. `orchestrator.handle_user` собирает **env_brief** и хвост истории.
2. **Junior** отдаёт `<ctrl>` (intent, tools, rag_query, neuro_update) + `<advice>` (стиль).
3. Применяем **нейропрофиль** → стиль/параметры генерации.
4. **RAG**: выдаём топ-факты (FAISS → e5 rerank).
5. **Senior** генерит строго **JSON** в `<json>…</json>`: `text`, опц. `tool_calls[]`, `memory[]`, `plan[]`.
6. Выполняются `tool_calls` → `tool_results` → мини-рефайн текста Senior.
7. **Guard** мягко маскирует PII/брань → сохраняем ответ и метаданные.

---

## Инструменты из коробки

* `note.create {text, tags?}` — заметка в SQLite
* `reminder.create {when, title, channel?}` — напоминание через APScheduler
* `messages.search_by_date {date, span_days?}` — выборка истории
* `alias.add {alias, short_desc?}` / `alias.set_primary {alias}` — алиасы
* `tg.message.send {to, text}` — опционально (если настроен `config/tg.yaml`)

Инструменты описаны в коде, вызываются строго через `tool_calls` Senior. Никаких внешних HTTP-запросов.

---

## Авто-сон (консолидация памяти)

```bash
python scripts/scheduler.py   # ежедневный запуск сна в 04:00
# или вручную:
python -m scripts.consolidate_sleep
```

* short → temp (дневные сводки, curated facts), temp → long по TTL
* переиндекс RAG, сброс нейропрофиля к базовым
* экспорт датасетов: SFT Senior и mini-LoRA Junior

---

## Тесты и самопроверка

```bash
# pytest-пакет: guard, neuro, bandit, rag, e2e с моками
pytest -q

# встроенная самодиагностика (готовность офлайн, запрет HTTP, модели, инструменты)
python -m scripts.self_check
```

---

## Частые проблемы и превентивные проверки

* **Невалидный JSON от Senior**
  Система делает несколько попыток и «чинит» ответ. Если совсем пусто — извинение коротким текстом.
  *Что улучшить:* держите `temperature > 0` при `do_sample=True`, либо `do_sample=False` для greedy.

* **`temperature (=0.0) has to be strictly positive`**
  Если хотите жёсткий greedy: ставьте `do_sample: false` **и уберите** `temperature`.

* **bitsandbytes без CUDA**
  Сообщение про «compiled without GPU support». Поставьте bnb, собранный под вашу CUDA, либо загрузите модель в 8-бит/FP16 без bnb.

* **Junior на CPU медленный**
  Это нормально для больших контекстов. Можно поднять `n_gpu_layers` (при CUDA-сборке `llama.cpp`) или взять ещё меньшую GGUF.

* **FAISS читает индекс каждый поиск**
  Индекс кэшируется в процессе. Если данных много — держите процесс живым (CLI/бот) и периодически «сон» переиндексирует.

* **Отключён интернет**
  Проект рассчитан на офлайн. Скачайте модели заранее (папки с весами/gguf в `config/llm.yaml`) — онлайн не нужен.

---

## Почему два агента

Junior дешёвый и быстрый: он **решает**, что делать (инструменты, RAG, стиль, настроение).
Senior сильный и дорогой: он **формулирует** финальный ответ и строго соблюдает формат/инструменты.
Так мы экономим токены/VRAM, повышаем управляемость и стабильность формата.

---

## Замены 

* **Senior**: можно заменить на Qwen-7B/8B, Llama-3.1-8B (если есть доступ), DeepSeek-R1-8B — просто укажите локальную папку весов и нужные параметры загрузки.
* **Junior**: любой 1–3B GGUF (Qwen/Gemma/Mistral-tiny), главное — корректный chat-template в промпте.
* **Embeddings**: переключаемся на `Qwen3-Embedding-0.6B` (чуть дороже, но лучше для RAG), либо оставляем `e5-small-v2`.

---

## Мини-FAQ

**Зачем Junior, если инструменты может вызывать Senior?**
Чтобы Senior не «гадал», нужен ли инструмент и какой — это решает Junior; Senior остаётся сфокусированным на хорошем тексте и строгом формате JSON.

**Насколько «живой» стиль?**
Очень: нейропрофиль (10 медиаторов) меняет температуру/длину/структурность/теплоту/юмор. Junior в `<ctrl>` может «подкрутить» уровни.

**Что с приватностью?**
Всё локально: модели, FAISS, SQLite, инструменты. Guard маскирует PII.

---

### Быстрый чек-лист перед запуском

* [ ] Указаны **локальные пути** к моделям в `config/llm.yaml`
  (Senior: папка Mistral-7B; Junior: GGUF-файл Gemma-3n-E4B)
* [ ] PyTorch соответствует вашей **CUDA**, bnb с CUDA (если нужен int4)
* [ ] `scripts.migrate` прошёл без ошибок
* [ ] (Опц.) `scripts.seed_rag` загрузил базовые тексты
* [ ] `python -m app.io.cli` — первый ответ может занять время (инициализация моделей)

---
В планах создания "сознания"(постоянно активная генерация мыслей вместо чат бота), упрощение системы общения между моделями, создание более простой системы расширения, создание более человеческого осознания, создание TTS и понимание голоса, а также собственный войс банк
