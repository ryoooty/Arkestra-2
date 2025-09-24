from app.core.router import rerank_e5


def test_rerank_basic():
    docs = [{"text": "добавили авто-сон"}, {"text": "junior style_directive"}, {"text": "e5 rerank RAG"}]
    rer = rerank_e5("rerank RAG", docs, top_k=2)
    assert len(rer) == 2
