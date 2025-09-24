from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.memory.db import get_conn, migrate, get_tool_instructions, upsert_bandit, add_feedback, mark_approved
from app.core.logs import log

app = FastAPI(title="Arkestra Admin API", version="1.0")


class ToolIn(BaseModel):
    name: str = Field(..., regex=r"^[a-z0-9._\-]+$")
    title: str
    description: str
    instruction: str
    entrypoint: str
    enabled: bool = True


class ToolUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    instruction: Optional[str] = None
    entrypoint: Optional[str] = None
    enabled: Optional[bool] = None


@app.on_event("startup")
def _startup():
    migrate()
    log.info("DB migrated. Admin API ready.")


# --- Tools CRUD ---
@app.get("/tools", response_model=List[Dict[str, Any]])
def list_tools():
    with get_conn() as c:
        cur = c.execute("SELECT name,title,description,instruction,entrypoint,enabled FROM tools ORDER BY name")
        return [dict(r) for r in cur.fetchall()]


@app.get("/tools/{name}")
def get_tool(name: str):
    with get_conn() as c:
        cur = c.execute("SELECT name,title,description,instruction,entrypoint,enabled FROM tools WHERE name=?", (name,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Tool not found")
        return dict(row)


@app.post("/tools", status_code=201)
def create_tool(t: ToolIn):
    with get_conn() as c:
        try:
            c.execute(
                "INSERT INTO tools(name,title,description,instruction,entrypoint,enabled) VALUES (?,?,?,?,?,?)",
                (t.name, t.title, t.description, t.instruction, t.entrypoint, 1 if t.enabled else 0),
            )
        except Exception as e:
            raise HTTPException(400, f"Insert failed: {e}")
    return {"ok": True, "name": t.name}


@app.put("/tools/{name}")
def update_tool(name: str, u: ToolUpdate):
    sets, vals = [], []
    for k, v in u.dict(exclude_unset=True).items():
        sets.append(f"{k}=?")
        vals.append(v if k != "enabled" else (1 if v else 0))
    if not sets:
        return {"ok": True}
    with get_conn() as c:
        c.execute(f"UPDATE tools SET {', '.join(sets)} WHERE name=?", (*vals, name))
        if c.rowcount == 0:
            raise HTTPException(404, "Tool not found")
    return {"ok": True}


@app.delete("/tools/{name}")
def delete_tool(name: str):
    with get_conn() as c:
        c.execute("DELETE FROM tools WHERE name=?", (name,))
        if c.rowcount == 0:
            raise HTTPException(404, "Tool not found")
    return {"ok": True}


# --- Environment facts (read) ---
@app.get("/env/{env_id}/facts")
def env_facts(env_id: int):
    with get_conn() as c:
        cur = c.execute(
            "SELECT key,value,importance,updated_at FROM env_facts WHERE env_id=? ORDER BY importance DESC, updated_at DESC",
            (env_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# --- Feedback / approve (writes) ---
class FeedbackIn(BaseModel):
    msg_id: int
    kind: str = Field(..., regex=r"^(up|down|text)$")
    text: Optional[str] = None
    intent: Optional[str] = None
    suggestion_kind: Optional[str] = None


@app.post("/feedback")
def submit_feedback(fb: FeedbackIn):
    add_feedback(fb.msg_id, fb.kind, fb.text)
    if fb.intent and fb.suggestion_kind:
        reward = 1 if fb.kind == "up" else (-1 if fb.kind == "down" else 0)
        if reward != 0:
            upsert_bandit(
                fb.intent,
                fb.suggestion_kind,
                wins_delta=1.0 if reward > 0 else 0.0,
                plays_delta=1.0,
            )
    return {"ok": True}


@app.post("/approve/{msg_id}")
def approve_msg(msg_id: int):
    mark_approved(msg_id, 1)
    return {"ok": True}


# --- Health / Metrics ---
@app.get("/health")
def health():
    return {"ok": True}


_METRICS = {
    "requests_total": 0,
    "errors_total": 0,
}


@app.middleware("http")
async def _metrics_mw(request, call_next):
    _METRICS["requests_total"] += 1
    try:
        resp = await call_next(request)
        return resp
    except Exception:
        _METRICS["errors_total"] += 1
        raise


@app.get("/metrics", response_class=None)
def metrics():
    # простейший Prometheus формат
    lines = [
        f'# HELP arkestra_requests_total total requests',
        f'# TYPE arkestra_requests_total counter',
        f'arkestra_requests_total {_METRICS["requests_total"]}',
        f'# HELP arkestra_errors_total total errors',
        f'# TYPE arkestra_errors_total counter',
        f'arkestra_errors_total {_METRICS["errors_total"]}',
    ]
    return "\n".join(lines)
