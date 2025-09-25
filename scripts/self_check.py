"""Self-diagnostics for the Arkestra local stack.

This script intentionally avoids any network activity and exercises the
critical components that power the CLI orchestrator flow. It runs a suite of
assertive checks while keeping the output readable via coloured PASS/FAIL
lines and a final summary.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple
import types

try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore
    HAS_NUMPY = False


def _import_core_modules() -> Dict[str, Any]:
    """Import project modules with graceful fallbacks for optional helpers."""

    imports: Dict[str, Any] = {}

    # Strict import order required by the checklist instructions.
    from app.core import neuro, guard, bandit, budget  # type: ignore

    imports.update({
        "neuro": neuro,
        "guard": guard,
        "bandit": bandit,
        "budget": budget,
    })

    from app.core.llm import generate as llm_generate  # type: ignore

    imports["llm_generate"] = llm_generate

    from app.core.reminders import init_scheduler, create_reminder, _fire  # type: ignore

    imports.update({
        "init_scheduler": init_scheduler,
        "create_reminder": create_reminder,
        "reminder_fire": _fire,
    })

    from app.core.orchestrator import handle_user  # type: ignore

    imports["handle_user"] = handle_user

    try:  # pragma: no cover - optional helper in repo
        from app.memory.db import migrate, set_db_path, get_conn  # type: ignore
    except ImportError:
        from app.memory import db as _db_module  # type: ignore

        migrate = _db_module.migrate
        get_conn = _db_module.get_conn

        def set_db_path(path: str) -> None:  # type: ignore
            _db_module.DB_PATH = Path(path)

        setattr(_db_module, "set_db_path", set_db_path)

    imports.update({
        "migrate": migrate,
        "set_db_path": set_db_path,
        "get_conn": get_conn,
    })

    from app.rag.encoders import encode, get_encoder_name  # type: ignore

    imports.update({
        "encode": encode,
        "get_encoder_name": get_encoder_name,
    })

    from app.rag import index as rag_index  # type: ignore
    from app.rag.index import add_texts, reset_index  # type: ignore

    imports.update({
        "rag_index": rag_index,
        "add_texts": add_texts,
        "reset_index": reset_index,
    })

    from app.tools import note as tool_note  # type: ignore
    from app.tools import reminder as tool_reminder  # type: ignore
    from app.tools import alias as tool_alias  # type: ignore
    from app.tools import search_by_date as tool_search  # type: ignore

    imports.update(
        {
            "tool_note": tool_note,
            "tool_reminder": tool_reminder,
            "tool_alias": tool_alias,
            "tool_search": tool_search,
        }
    )

    return imports


def _ensure_tiktoken_stub() -> None:
    try:
        import tiktoken  # type: ignore
    except ModuleNotFoundError:
        stub = types.ModuleType("tiktoken")

        class _SimpleEncoding:
            def encode(self, text: str) -> List[int]:
                return [len(token) for token in text.split()] or [len(text)]

        def encoding_for_model(model: str) -> _SimpleEncoding:
            return _SimpleEncoding()

        def get_encoding(name: str) -> _SimpleEncoding:
            return _SimpleEncoding()

        stub.encoding_for_model = encoding_for_model  # type: ignore[attr-defined]
        stub.get_encoding = get_encoding  # type: ignore[attr-defined]
        sys.modules["tiktoken"] = stub


def _ensure_yaml_stub() -> None:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        from app.util import simple_yaml

        stub = types.ModuleType("yaml")

        def safe_load(text: str) -> Any:
            return simple_yaml.loads(text)

        def safe_dump(data: Any, **_: Any) -> str:
            return json.dumps(data)

        stub.safe_load = safe_load  # type: ignore[attr-defined]
        stub.safe_dump = safe_dump  # type: ignore[attr-defined]
        sys.modules["yaml"] = stub


_ensure_tiktoken_stub()
_ensure_yaml_stub()
def _ensure_apscheduler_stub() -> None:
    try:
        import apscheduler  # type: ignore
    except ModuleNotFoundError:
        sched_module = types.ModuleType("apscheduler")
        submodule = types.ModuleType("apscheduler.schedulers.background")

        class BackgroundScheduler:
            def __init__(self, daemon: bool = True) -> None:
                self.daemon = daemon
                self.jobs: List[Dict[str, Any]] = []

            def start(self) -> None:
                return None

            def add_job(
                self,
                func: Callable,
                trigger: str,
                run_date: Any = None,
                args: List[Any] | Tuple[Any, ...] | None = None,
                id: str | None = None,
                replace_existing: bool | None = None,
            ) -> None:
                self.jobs.append(
                    {
                        "func": func,
                        "trigger": trigger,
                        "run_date": run_date,
                        "args": list(args or []),
                        "id": id,
                        "replace_existing": replace_existing,
                    }
                )

        submodule.BackgroundScheduler = BackgroundScheduler  # type: ignore[attr-defined]
        sys.modules["apscheduler"] = sched_module
        sys.modules["apscheduler.schedulers"] = types.ModuleType("apscheduler.schedulers")
        sys.modules["apscheduler.schedulers.background"] = submodule


_ensure_apscheduler_stub()

MODS = _import_core_modules()


# After the imports are available enforce the "no HTTP" guardrails.
FORBIDDEN_MODULES = ("requests", "fastapi", "uvicorn")
for _mod in FORBIDDEN_MODULES:
    if _mod in sys.modules:
        raise RuntimeError(f"no-HTTP policy violated: '{_mod}' module is loaded")


class Colours:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    @classmethod
    def enable(cls) -> bool:
        if os.name == "nt":
            return False
        return sys.stdout.isatty()


COLOURS_ENABLED = Colours.enable()


def _c(text: str, colour: str) -> str:
    if not COLOURS_ENABLED:
        return text
    return f"{colour}{text}{Colours.RESET}"


StepFunc = Callable[[], Dict[str, Any] | None]


def run_step(name: str, func: StepFunc, ctx: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    summary: Dict[str, Any] = {"name": name, "status": "PASS"}
    try:
        result = func() or {}
        summary.update(result)
    except Exception as exc:  # pragma: no cover - defensive by design
        summary["status"] = "FAIL"
        hint = ctx.setdefault("hints", [])
        hint_msg = _build_hint(name, exc)
        hint.append(hint_msg)
        tb_lines = traceback.format_exc().splitlines()
        detail = tb_lines[-1] if tb_lines else str(exc)
        summary["message"] = detail
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        summary["elapsed_ms"] = elapsed_ms

    status = summary.get("status", "PASS")
    message = summary.get("message", "")
    prefix = "[PASS]"
    colour = Colours.GREEN
    if status == "WARN":
        prefix = "[WARN]"
        colour = Colours.YELLOW
    elif status == "FAIL":
        prefix = "[FAIL]"
        colour = Colours.RED

    text = f"{prefix} {name}"
    text += f" (dt={summary['elapsed_ms']:.1f} ms)"
    if message:
        text += f": {message}"

    print(_c(text, colour))
    return summary


def _build_hint(step: str, exc: Exception) -> str:
    msg = str(exc)
    lower = msg.lower()
    if "ggml" in lower or "llama" in lower or "model" in lower:
        return (
            f"{step}: убедитесь, что путь к junior модели прописан в config/llm.yaml "
            "(параметр junior.model_path)."
        )
    if "no module named 'torch'" in lower or "torch" in lower:
        return (
            f"{step}: установите PyTorch/transformers (pip install torch transformers --extra-index-url https://download.pytorch.org/whl/cpu)."
        )
    if "cuda" in lower or "device" in lower:
        return (
            f"{step}: проверьте сборку PyTorch с поддержкой CUDA (torch.version +cuXXX) "
            "или используйте CPU-конфигурацию."
        )
    if "numpy" in lower:
        return (
            f"{step}: установите зависимость numpy (pip install numpy или pip install -r requirements.txt)."
        )
    if "dim" in lower or "faiss" in lower:
        return (
            f"{step}: похоже на несоответствие размерности RAG-индекса — удалите каталог "
            "data/rag и создайте его заново."
        )
    if "rowcount" in lower and "sqlite" in lower:
        return (
            f"{step}: обновите инструмент alias.set_primary — используйте cursor.rowcount вместо connection.rowcount."
        )
    return f"{step}: {msg}"


def main() -> int:
    ctx: Dict[str, Any] = {"hints": [], "warnings": []}
    pass_count = 0
    warn_count = 0
    fail_count = 0

    def _record(summary: Dict[str, Any]) -> None:
        nonlocal pass_count, warn_count, fail_count
        status = summary.get("status", "PASS")
        if status == "FAIL":
            fail_count += 1
        elif status == "WARN":
            warn_count += 1
            ctx.setdefault("warnings", []).append(summary.get("message", ""))
        else:
            pass_count += 1

    # Step 1: data directories / DB isolation
    data_root_holder: Dict[str, Path] = {}

    def step_data_dirs() -> Dict[str, Any]:
        raw_root = os.getenv("ARKESTRA_DATA_DIR")
        if raw_root:
            data_root = Path(raw_root).expanduser().resolve()
            data_root.mkdir(parents=True, exist_ok=True)
        else:
            data_root = Path(tempfile.mkdtemp(prefix="arkestra_selfcheck_"))
        print(_c(f"Data root: {data_root}", Colours.CYAN))
        data_root_holder["root"] = data_root

        db_path = data_root / "arkestra.sqlite"
        MODS["set_db_path"](str(db_path))

        rag_dir = data_root / "rag"
        rag_dir.mkdir(parents=True, exist_ok=True)

        rag_index = MODS["rag_index"]
        rag_index.DATA_DIR = rag_dir
        rag_index.INDEX_PATH = rag_dir / "faiss.index"
        rag_index.ROWS_PATH = rag_dir / "rows.jsonl"
        rag_index.INFO_PATH = rag_dir / "meta.json"

        MODS["reset_index"]()

        # Ensure sqlite WAL/SHM files cleaned on temp dir reuse
        for pattern in ("arkestra.sqlite-wal", "arkestra.sqlite-shm"):
            wal_path = data_root / pattern
            if wal_path.exists():
                wal_path.unlink()

        ctx["data_root"] = data_root
        return {"message": "data directories prepared"}

    # Step 2: migrate DB
    def step_db_migrate() -> Dict[str, Any]:
        MODS["migrate"]()
        with MODS["get_conn"]() as conn:
            for table in ("messages", "facts", "aliases"):
                conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
        return {"message": "DB migrate"}

    # Step 3: RAG encoder + FAISS
    def step_rag() -> Dict[str, Any]:
        encoder_name = MODS["get_encoder_name"]()
        message = "RAG add_texts"
        status = "PASS"
        if encoder_name != "qwen3-0.6b":
            status = "WARN"
            message = f"Ожидался encoder 'qwen3-0.6b', но получен '{encoder_name}'"

        if not HAS_NUMPY:
            raise AssertionError("NumPy не установлен — pip install numpy или pip install -r requirements.txt")

        texts = [
            "Мы тестируем RAG индекс.",
            "Аркестра — локальная ИИ-архитектура.",
            "Необязательная строка.",
        ]
        X = MODS["encode"](texts)
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float32)
        assert X.shape[0] == len(texts), "encoder returned unexpected row count"
        assert X.dtype == np.float32, "encoder must emit float32 vectors"
        assert X.flags["C_CONTIGUOUS"], "encoder output must be contiguous"

        MODS["add_texts"](X, encoder_name=encoder_name)
        MODS["add_texts"](X, encoder_name=encoder_name)

        return {"status": status, "message": message}

    # Step 4: Neuro checks
    def step_neuro() -> Dict[str, Any]:
        neuro = MODS["neuro"]
        snap = neuro.snapshot()
        expected_keys = {
            "dopamine",
            "serotonin",
            "norepinephrine",
            "acetylcholine",
            "gaba",
            "glutamate",
            "endorphins",
            "oxytocin",
            "vasopressin",
            "histamine",
        }
        assert set(snap.keys()) == expected_keys, "neuro snapshot missing keys"

        preset = neuro.bias_to_style()
        temp = float(preset.get("temperature", 0.0))
        max_tokens = int(preset.get("max_tokens", 0))
        assert 0.1 <= temp <= 1.3, "temperature out of expected range"
        assert 128 <= max_tokens <= 1024, "max_tokens out of expected range"

        neuro.apply_delta({"dopamine": +2, "gaba": +1})
        preset2 = neuro.bias_to_style()
        assert preset2 != preset, "neuro delta had no effect"

        neuro.sleep_reset()
        snap2 = neuro.snapshot()
        assert snap2 == snap, "neuro sleep_reset did not restore baseline"
        return {"message": "neuro preset"}

    # Step 5: Guard
    def step_guard() -> Dict[str, Any]:
        guard = MODS["guard"]
        s = "Это сука тест. Почта test@example.com, телефон +7 900 123-45-67."
        masked, hits = guard.soft_censor(s)
        assert hits["profanity"] >= 1, "profanity mask did not trigger"
        assert "***" in masked, "masking indicator missing"
        assert "скрыт" in masked, "PII masking marker missing"
        return {"message": "guard soft_censor"}

    # Step 6: Bandit
    def step_bandit() -> Dict[str, Any]:
        bandit = MODS["bandit"]
        suggestions = [
            {"intent": "task", "kind": "good", "confidence": 0.6},
            {"intent": "task", "kind": "mischief", "confidence": 0.4},
        ]
        pick = bandit.pick("task", suggestions)
        assert pick in suggestions, "bandit pick returned unknown suggestion"
        bandit.update("task", pick.get("kind", "good"), +1)
        return {"message": "bandit pick/update"}

    # Step 7: Junior LLM
    def step_junior() -> Dict[str, Any]:
        generator = MODS["llm_generate"]
        prompt = (
            "Верни JSON v2: intent, tools_hint[], style_directive, neuro_update, rag_query? — строго JSON."
        )
        out = generator(
            "junior",
            prompt,
            max_new_tokens=64,
            temperature=0.2,
            stop=["\n\n", "```"],
        )
        assert isinstance(out, str), "junior generate must return string"

        text = out.strip()
        parsed: Dict[str, Any]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if text.startswith("{") and text.endswith("}"):
                text2 = text.replace("'", '"')
                parsed = json.loads(text2)
            else:
                raise AssertionError(
                    "junior output is not JSON; проверьте модель и prompt"
                )
        required_keys = {"intent", "style_directive"}
        assert required_keys.issubset(parsed), "junior JSON missing required keys"
        return {"message": "junior generate"}

    # Step 8: Senior LLM
    def step_senior() -> Dict[str, Any]:
        generator = MODS["llm_generate"]
        prompt = "Скажи 'ok' одним словом."
        out = generator(
            "senior",
            prompt,
            max_new_tokens=16,
            temperature=0.3,
            stop=["\n\n"],
        )
        assert isinstance(out, str), "senior generate must return string"
        if "ok" not in out.lower() and not out.strip():
            raise AssertionError("senior output empty; проверьте модель и GPU")
        return {"message": "senior generate"}

    # Step 9: Tools — note
    def step_note_tool() -> Dict[str, Any]:
        tool_note = MODS["tool_note"]
        create_func = getattr(tool_note, "create", None)
        if create_func is None:
            create_func = getattr(tool_note, "main")

        payload = {
            "user_id": 1,
            "text": "небо красное",
            "tags": ["test"],
        }
        if create_func is tool_note.main:
            result = create_func(payload)
        else:
            result = create_func(**payload)
        assert result.get("ok"), "note.create returned failure"

        with MODS["get_conn"]() as conn:
            cur = conn.execute(
                "SELECT text FROM notes WHERE text=? ORDER BY id DESC LIMIT 1",
                (payload["text"],),
            )
            row = cur.fetchone()
            assert row is not None, "note not persisted"
        return {"message": "tool note.create"}

    # Step 10: Tools — reminder
    def step_reminder_tool() -> Dict[str, Any]:
        MODS["init_scheduler"]()
        tool_reminder = MODS["tool_reminder"]
        create_func = getattr(tool_reminder, "create", None)
        if create_func is None:
            create_func = getattr(tool_reminder, "main")

        when_dt = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=2)
        when_iso = when_dt.isoformat().replace("+00:00", "Z")
        payload = {
            "user_id": 1,
            "title": "test reminder",
            "when": when_iso,
            "channel": "cli",
        }

        def _invoke(p: Dict[str, Any]):
            if create_func is tool_reminder.main:
                return create_func(p)
            return create_func(**p)

        result = _invoke(payload)
        if not result.get("ok") and "format" in str(result.get("error")):
            dt = _dt.datetime.fromisoformat(when_iso.replace("Z", ""))
            payload["when"] = dt.strftime("%Y-%m-%d %H:%M")
            result = _invoke(payload)

        assert result.get("ok"), f"reminder.create failed: {result}"

        rem_id = result.get("reminder_id") or result.get("id")
        if rem_id is None:
            with MODS["get_conn"]() as conn:
                cur = conn.execute(
                    "SELECT id FROM reminders WHERE title=? ORDER BY id DESC LIMIT 1",
                    (payload["title"],),
                )
                row = cur.fetchone()
                rem_id = int(row["id"]) if row else None
        assert rem_id is not None, "reminder id not available"

        MODS["reminder_fire"](int(rem_id), payload["title"], payload["channel"])

        with MODS["get_conn"]() as conn:
            cur = conn.execute(
                "SELECT text FROM messages WHERE role='assistant' AND text LIKE ? ORDER BY id DESC LIMIT 1",
                ("%test reminder%",),
            )
            row = cur.fetchone()
            assert row is not None, "reminder fire did not log message"
        return {"message": "tool reminder.create"}

    # Step 11: Tools — alias
    def step_alias_tool() -> Dict[str, Any]:
        tool_alias = MODS["tool_alias"]
        add_res = tool_alias.add({"user_id": 1, "alias": "кот", "short_desc": "ласковое"})
        assert add_res.get("ok"), f"alias.add failed: {add_res}"
        primary_res = tool_alias.set_primary({"user_id": 1, "alias": "кот"})
        assert primary_res.get("ok"), f"alias.set_primary failed: {primary_res}"
        return {"message": "alias add/set_primary"}

    # Step 12: Tools — search_by_date
    def step_search_tool() -> Dict[str, Any]:
        tool_search = MODS["tool_search"]
        today = _dt.date.today().isoformat()
        payload = {"user_id": 1, "date": today, "span_days": 1, "limit": 50}
        search_func = getattr(tool_search, "search", None)
        if search_func is None:
            search_func = getattr(tool_search, "main")
        if search_func is tool_search.main:
            res = search_func(payload)
        else:
            res = search_func(**payload)
        assert res.get("ok"), f"search_by_date failed: {res}"
        assert isinstance(res.get("items"), list), "search_by_date items must be list"
        return {"message": "search_by_date"}

    # Step 13: Orchestrator end-to-end
    def step_orchestrator() -> Dict[str, Any]:
        res1 = MODS["handle_user"](
            user_id="1",
            text="сохрани в заметки: тестовая заметка",
            channel="cli",
            chat_id="local",
        )
        assert isinstance(res1, dict) or res1 is None, "handle_user returned invalid type"

        res2 = MODS["handle_user"](
            user_id="1",
            text="сделай напоминание через минуту с текстом 'привет себе'",
            channel="cli",
            chat_id="local",
        )
        assert isinstance(res2, dict) or res2 is None, "handle_user second call failed"
        return {"message": "orchestrator"}

    # Step 14: Budget pack
    def step_budget() -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            history.append(
                {
                    "role": role,
                    "text": "".join(["Lorem ipsum dolor sit amet "] * 10),
                }
            )
        rag_hits = [
            {"text": "RAG hit example", "score": 0.9},
            {"text": "Another hit", "score": 0.5},
        ]
        junior_meta = {"intent": "test", "tools_hint": []}
        preset = MODS["neuro"].bias_to_style()
        max_tokens = int(preset.get("max_tokens", 512))
        packed = MODS["budget"].trim(history, rag_hits, junior_meta, max_tokens)
        assert isinstance(packed, dict)
        hist = packed.get("history", [])
        assert len(hist) <= len(history)
        # approximate token budget validation
        approx_tokens = 0
        for item in hist:
            approx_tokens += len(str(item.get("text", "")).split())
        approx_tokens += sum(len(h.get("text", "").split()) for h in packed.get("rag_hits", []))
        approx_tokens += len(str(packed.get("junior_meta", {}))) // 4
        assert approx_tokens <= max_tokens * 1.5, "budget trim exceeded limits"
        return {"message": "budget trim"}

    # Step 15: No-HTTP police
    def step_no_http() -> Dict[str, Any]:
        llm_path = Path("app/core/llm.py")
        if not llm_path.exists():
            raise FileNotFoundError("app/core/llm.py not found")
        text = llm_path.read_text(encoding="utf-8")
        hints: List[str] = []
        for marker in ("requests.post", "http://", "https://", "endpoint"):
            if marker in text:
                line_no = _find_line_number(text, marker)
                hints.append(f"{marker} @ line {line_no}")
        if hints:
            raise AssertionError("no-HTTP policy violated in app/core/llm.py: " + ", ".join(hints))
        return {"message": "no-HTTP policy"}

    steps: Iterable[Tuple[str, StepFunc]] = [
        ("data dirs", step_data_dirs),
        ("DB migrate", step_db_migrate),
        ("RAG encoder", step_rag),
        ("neuro", step_neuro),
        ("guard", step_guard),
        ("bandit", step_bandit),
        ("junior LLM", step_junior),
        ("senior LLM", step_senior),
        ("tool note", step_note_tool),
        ("tool reminder", step_reminder_tool),
        ("tool alias", step_alias_tool),
        ("tool search", step_search_tool),
        ("orchestrator", step_orchestrator),
        ("budget", step_budget),
        ("no-HTTP", step_no_http),
    ]

    for name, func in steps:
        summary = run_step(name, func, ctx)
        _record(summary)

    total = pass_count + warn_count + fail_count
    summary_line = (
        f"Summary: {pass_count} pass, {warn_count} warn, {fail_count} fail out of {total} checks"
    )
    colour = Colours.GREEN if fail_count == 0 else Colours.RED
    print(_c("=" * len(summary_line), colour))
    print(_c(summary_line, colour))
    print(_c("=" * len(summary_line), colour))

    if ctx.get("warnings"):
        print(_c("Warnings:", Colours.YELLOW))
        for w in ctx["warnings"]:
            if w:
                print(" -", w)

    if ctx.get("hints"):
        print(_c("Hints:", Colours.CYAN))
        for hint in ctx["hints"]:
            print(" -", hint)

    data_root = data_root_holder.get("root")
    if data_root and not os.getenv("ARKESTRA_DATA_DIR"):
        try:
            shutil.rmtree(data_root, ignore_errors=True)
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    return 0 if fail_count == 0 else 1


def _find_line_number(text: str, needle: str) -> int:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return -1


if __name__ == "__main__":
    sys.exit(main())

