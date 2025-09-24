from app.core.orchestrator import handle_user
import app.core.llm as llm
from app.core import tools_runner
from app.memory import db
from tests.mock_llm import junior_mock, senior_mock


def test_orchestrator_e2e(monkeypatch, tmp_path):
    test_db = tmp_path / "arkestra.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)
    db.migrate()

    def _mock_generate(role, prompt, **kw):
        return (
            junior_mock(role, prompt, **kw)
            if role == "junior"
            else senior_mock(role, prompt, **kw)
        )

    monkeypatch.setattr(llm, "generate", _mock_generate)
    monkeypatch.setattr("app.agents.junior.llm_generate", _mock_generate)
    monkeypatch.setattr("app.agents.senior.llm_generate", _mock_generate)

    monkeypatch.setattr(
        tools_runner,
        "run_all",
        lambda calls: [
            {"name": call["name"], "result": {"ok": True, **call.get("args", {})}}
            for call in (calls or [])
        ],
    )

    res = handle_user("u1", "создай заметку о сегодняшней работе")
    assert "замет" in res["text"].lower()
    assert res["tool_calls"][0]["name"] == "note.create"
