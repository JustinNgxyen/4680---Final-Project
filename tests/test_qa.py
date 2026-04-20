from fastapi.testclient import TestClient
import app.main as main_module

client = TestClient(main_module.app)

class FakeLLM:
    async def generate(self, messages):
        # deterministic
        return "Recursion is when a function calls itself."

def test_qa_returns_citations_and_turn_count(monkeypatch):
    # reset store
    main_module.STORE.chunks.clear()
    main_module.STORE.doc_chunks.clear()
    main_module.STORE.sessions.clear()

    # ingest something relevant
    text = "Recursion is when a function calls itself. It solves problems by reducing them."
    r = client.post("/ingest", json={"doc_id": "intro_cs_notes", "text": text})
    assert r.status_code == 200

    # patch llm
    monkeypatch.setattr(main_module, "get_llm_client", lambda: FakeLLM())
    # lower threshold for deterministic embedder in tests
    monkeypatch.setattr(main_module, "OUT_OF_SCOPE_THRESHOLD", 0.0)

    payload = {"session_id": "s1", "question": "What is recursion?", "k": 4}
    q = client.post("/qa", json=payload)
    assert q.status_code == 200
    data = q.json()

    assert data["answer"] == "Recursion is when a function calls itself."
    assert isinstance(data["citations"], list)
    assert len(data["citations"]) >= 1
    assert "chunk_id" in data["citations"][0]
    assert "score" in data["citations"][0]
    assert data["turn_count"] == 1

    # second question increments turn_count
    q2 = client.post("/qa", json={"session_id": "s1", "question": "Explain recursion again.", "k": 2})
    assert q2.status_code == 200
    assert q2.json()["turn_count"] == 2

def test_qa_out_of_scope(monkeypatch):
    main_module.STORE.chunks.clear()
    main_module.STORE.doc_chunks.clear()
    main_module.STORE.sessions.clear()

    # ingest unrelated doc
    r = client.post("/ingest", json={"doc_id": "d1", "text": "FastAPI uses Pydantic for validation."})
    assert r.status_code == 200

    # force out-of-scope by using high threshold
    monkeypatch.setattr(main_module, "OUT_OF_SCOPE_THRESHOLD", 0.99)

    q = client.post("/qa", json={"session_id": "s2", "question": "What is recursion?", "k": 3})
    assert q.status_code == 200
    data = q.json()
    assert data["citations"] == []
    assert "don't know" in data["answer"].lower()