from fastapi.testclient import TestClient
from app.main import app, STORE


client = TestClient(app)


def test_ingest_replaces_doc_chunks():
    # reset store for test isolation
    STORE.chunks.clear()
    STORE.doc_chunks.clear()

    doc_id = "intro_cs_notes"
    text = " ".join(["hello"] * 500)  # enough words for multiple chunks with defaults

    r = client.post("/ingest", json={"doc_id": doc_id, "text": text})
    assert r.status_code == 200
    data = r.json()
    assert data["doc_id"] == doc_id
    assert data["chunks_added"] > 1

    first_count = STORE.count_chunks_for_doc(doc_id)

    # ingest again; should replace
    r2 = client.post("/ingest", json={"doc_id": doc_id, "text": " ".join(["world"] * 500)})
    assert r2.status_code == 200
    assert STORE.count_chunks_for_doc(doc_id) == first_count