from fastapi.testclient import TestClient
from app.main import app, STORE

client = TestClient(app)

def test_ingest_then_search_returns_results():
    STORE.chunks.clear()
    STORE.doc_chunks.clear()

    text = "FastAPI uses Pydantic for validation. " * 200
    r = client.post("/ingest", json={"doc_id": "d1", "text": text})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1

    s = client.get("/search", params={"query": "What does FastAPI use for validation?", "k": 3})
    assert s.status_code == 200
    data = s.json()

    assert data["query"]
    assert len(data["results"]) >= 1
    assert "chunk_id" in data["results"][0]
    assert "score" in data["results"][0]
    assert "text" in data["results"][0]