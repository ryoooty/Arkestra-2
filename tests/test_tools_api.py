from fastapi.testclient import TestClient

from app.memory import db
from app.server.main import app


def test_tools_crud(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "tools.db", raising=False)

    client = TestClient(app)

    tool = {
        "name": "test.tool",
        "title": "Test",
        "description": "desc",
        "instruction": "call with args",
        "entrypoint": "app.tools.test:main",
        "enabled": True,
    }

    response = client.post("/tools", json=tool)
    assert response.status_code == 201

    response = client.get("/tools")
    assert response.status_code == 200
    tools = response.json()
    assert any(item["name"] == "test.tool" for item in tools)
