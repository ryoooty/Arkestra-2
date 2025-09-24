from fastapi.testclient import TestClient

from app.server.main import ChatIn, app


def test_chat_endpoint_stub(monkeypatch):
    c = TestClient(app)

    # monkeypatch orchestrator
    def fake_handle_user(user_id, text, channel, chat_id, participants=None):
        return {"text": "stub reply", "tool_calls": [], "rag_hits": []}

    monkeypatch.setattr("app.server.main.handle_user", fake_handle_user)

    payload = ChatIn(user_id="u1", text="hello")
    r = c.post("/chat", json=payload.dict())
    assert r.status_code == 200
    j = r.json()
    assert j["text"] == "stub reply"
