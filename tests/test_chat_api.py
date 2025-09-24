from fastapi.testclient import TestClient

from app.memory import db
from app.server.main import app


def test_chat_endpoint_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "chat.db", raising=False)

    client = TestClient(app)

    def fake_handle_user(user_id, text, channel, chat_id, participants=None):
        return {"text": "stub reply", "tool_calls": [], "rag_hits": []}

    monkeypatch.setattr("app.server.main.handle_user", fake_handle_user)

    response = client.post("/chat", json={"user_id": "u1", "text": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "stub reply"
