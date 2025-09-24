from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.memory.db import get_conn, migrate


class Tool(BaseModel):
    name: str
    title: str
    description: str = ""
    instruction: str = ""
    entrypoint: str
    enabled: bool = True


class ToolIn(Tool):
    pass


class ToolUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    instruction: Optional[str] = None
    entrypoint: Optional[str] = None
    enabled: Optional[bool] = None


app = FastAPI(title="Arkestra Server")
templates = Jinja2Templates(directory="app/server/templates")


@app.on_event("startup")
def on_startup() -> None:
    migrate()


def _row_to_tool(row: sqlite3.Row) -> Tool:
    return Tool(
        name=row["name"],
        title=row["title"],
        description=row["description"] or "",
        instruction=row["instruction"] or "",
        entrypoint=row["entrypoint"],
        enabled=bool(row["enabled"]),
    )


def list_tools() -> list[Tool]:
    with get_conn() as c:
        cur = c.execute(
            "SELECT name,title,description,instruction,entrypoint,enabled FROM tools ORDER BY name"
        )
        return [_row_to_tool(row) for row in cur.fetchall()]


def get_tool(name: str) -> Tool:
    with get_conn() as c:
        cur = c.execute(
            "SELECT name,title,description,instruction,entrypoint,enabled FROM tools WHERE name=?",
            (name,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="tool not found")
        return _row_to_tool(row)


def create_tool(payload: ToolIn) -> Tool:
    with get_conn() as c:
        try:
            c.execute(
                "INSERT INTO tools(name,title,description,instruction,entrypoint,enabled) VALUES (?,?,?,?,?,?)",
                (
                    payload.name,
                    payload.title,
                    payload.description,
                    payload.instruction,
                    payload.entrypoint,
                    int(payload.enabled),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="tool already exists") from exc
    return get_tool(payload.name)


def update_tool(name: str, payload: ToolUpdate) -> Tool:
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        return get_tool(name)
    columns: list[str] = []
    values: list[object] = []
    for key, value in updates.items():
        if key == "enabled" and value is not None:
            value = int(value)
        columns.append(f"{key}=?")
        values.append(value)
    values.append(name)
    with get_conn() as c:
        cur = c.execute(
            f"UPDATE tools SET {', '.join(columns)} WHERE name=?",
            values,
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="tool not found")
    return get_tool(name)


def delete_tool(name: str) -> None:
    with get_conn() as c:
        cur = c.execute("DELETE FROM tools WHERE name=?", (name,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="tool not found")


@app.get("/tools", response_model=list[Tool])
def api_list_tools() -> list[Tool]:
    return list_tools()


@app.get("/tools/{name}", response_model=Tool)
def api_get_tool(name: str) -> Tool:
    return get_tool(name)


@app.post("/tools", response_model=Tool, status_code=201)
def api_create_tool(tool: ToolIn) -> Tool:
    return create_tool(tool)


@app.put("/tools/{name}", response_model=Tool)
def api_update_tool(name: str, tool: ToolUpdate) -> Tool:
    return update_tool(name, tool)


@app.delete("/tools/{name}", status_code=204)
def api_delete_tool(name: str) -> Response:
    delete_tool(name)
    return Response(status_code=204)


@app.get("/ui/tools", response_class=HTMLResponse)
def ui_tools(request: Request):
    tools = list_tools()
    return templates.TemplateResponse("tools.html", {"request": request, "tools": tools})


@app.post("/ui/tools")
def ui_create_tool(
    request: Request,
    name: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    instruction: str = Form(""),
    entrypoint: str = Form(...),
    enabled: Optional[str] = Form(None),
):
    create_tool(
        ToolIn(
            name=name,
            title=title,
            description=description,
            instruction=instruction,
            entrypoint=entrypoint,
            enabled=bool(enabled),
        )
    )
    return RedirectResponse(url="/ui/tools", status_code=303)


@app.get("/ui/tools/{name}", response_class=HTMLResponse)
def ui_edit_tool(request: Request, name: str):
    tool = get_tool(name)
    return templates.TemplateResponse("tool_edit.html", {"request": request, "tool": tool})


@app.post("/ui/tools/{name}")
def ui_update_tool(
    request: Request,
    name: str,
    title: str = Form(""),
    description: str = Form(""),
    instruction: str = Form(""),
    entrypoint: str = Form(""),
    enabled: Optional[str] = Form(None),
):
    update_tool(
        name,
        ToolUpdate(
            title=title,
            description=description,
            instruction=instruction,
            entrypoint=entrypoint,
            enabled=bool(enabled),
        ),
    )
    return RedirectResponse(url=f"/ui/tools/{name}", status_code=303)


@app.post("/ui/tools/{name}/delete")
def ui_delete_tool(name: str):
    delete_tool(name)
    return RedirectResponse(url="/ui/tools", status_code=303)
