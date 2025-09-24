# tests/mock_llm.py

def junior_mock(role: str, prompt: str, **kw) -> str:
    # Возвращаем валидный JSON v2
    return (
        '{"intent":"task","tools_hint":["note.create"],'
        '"style_directive":"коротко и тепло",'
        '"neuro_update":{"levels":{"dopamine":8,"serotonin":6}},'
        '"rag_query":"что мы делали сегодня"}'
    )


def senior_mock(role: str, prompt: str, **kw) -> str:
    # Если есть TOOL_INSTRUCTIONS — вызовем note.create
    return '{"text":"Ок, сохранила заметку.","tool_calls":[{"name":"note.create","args":{"text":"Мы работали над RAG и авто-сном"}}]}'
