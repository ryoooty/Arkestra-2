from fastapi.testclient import TestClient

from app.memory import db
from app.server.main import ToolIn, app


def test_tools_crud(tmp_path, monkeypatch):
    db_path = tmp_path / "tools.db"
    monkeypatch.setattr(db, "DB_PATH", db_path, raising=False)
    db.migrate()

    c = TestClient(app)
    # create tool
    t = ToolIn(
        name="test.tool",
        title="Test",
        description="desc",
        instruction="call with args",
        entrypoint="app.tools.test:main",
        enabled=True,
    )
    r = c.post("/tools", t=t)
    assert r.status_code == 201
    # list
    r = c.get("/tools")
    assert any(x["name"] == "test.tool" for x in r.json())
