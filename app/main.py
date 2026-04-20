from __future__ import annotations
import re

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .models import IngestRequest, IngestResponse, ChunkRecord, QARequest, QAResponse, Citation
from .store import InMemoryStore
from .chunking import chunk_text_words
from .embeddings import get_embeddings_client
from .similarity import cosine_similarity
from .retrieval import retrieve_top_k
from .qa_prompt import build_grounded_prompt
from .llm import get_llm_client

from .models import AgentRequest, AgentResponse, Step
from tools.calculator import eval_expr
from tools.knowledge import search_knowledge
from agent.decision import decide_action
from .llm import get_llm_client

app = FastAPI(title="Mini RAG Server", version="0.1.0")

# In-memory singleton store (base version requirement)
STORE = InMemoryStore()

# Config (can be env-driven later)
DEFAULT_CHUNK_SIZE_WORDS = 200
DEFAULT_OVERLAP_WORDS = 40
OUT_OF_SCOPE_THRESHOLD = 0.10

class IngestOptions(BaseModel):
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS
    overlap: int = DEFAULT_OVERLAP_WORDS

class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE_WORDS, ge=1),
    overlap: int = Query(DEFAULT_OVERLAP_WORDS, ge=0),
):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    if overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="overlap must be < chunk_size")

    chunk_texts = chunk_text_words(text, chunk_size=chunk_size, overlap=overlap)
    ...

    if not chunk_texts:
        raise HTTPException(status_code=400, detail="No chunks produced")

    embeddings_client = get_embeddings_client()
    embeddings = await embeddings_client.embed_texts(chunk_texts)

    if len(embeddings) != len(chunk_texts):
        raise HTTPException(status_code=500, detail="Embeddings count mismatch")

    chunk_records: list[ChunkRecord] = []
    for i, (ctext, emb) in enumerate(zip(chunk_texts, embeddings)):
        chunk_id = f"{req.doc_id}#{i}"
        chunk_records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=req.doc_id,
                chunk_index=i,
                text=ctext,
                embedding=emb,
            )
        )

    chunks_added = STORE.upsert_document_chunks(req.doc_id, chunk_records)
    return IngestResponse(doc_id=req.doc_id, chunks_added=chunks_added)

@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1),
    k: int = Query(3, ge=1, le=20),
):
    # Ensure we have data
    if not STORE.chunks:
        return SearchResponse(query=query, results=[])

    embeddings_client = get_embeddings_client()
    q_emb = (await embeddings_client.embed_texts([query]))[0]

    scored: list[tuple[str, float, str]] = []
    for chunk_id, chunk in STORE.chunks.items():
        score = cosine_similarity(q_emb, chunk.embedding)
        scored.append((chunk_id, float(score), chunk.text))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    results = [SearchResult(chunk_id=cid, score=s, text=txt) for cid, s, txt in top]
    return SearchResponse(query=query, results=results)

@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    # 1) Retrieve
    retrieved = await search_knowledge(STORE, req.query, k=4)

    # 2) Out-of-scope handling: if nothing retrieved or all scores too low
    if not retrieved or (retrieved and retrieved[0][1] < OUT_OF_SCOPE_THRESHOLD):
        # still append user turn for history
        STORE.append_message(req.session_id, "user", req.question)
        answer = "I don't know based on the documents I've been given."
        STORE.append_message(req.session_id, "assistant", answer)
        turn_count = len(STORE.get_history(req.session_id)) // 2
        return QAResponse(answer=answer, citations=[], turn_count=turn_count)

    # 3) Build grounded prompt with history
    history = STORE.get_history(req.session_id)
    prompt = build_grounded_prompt(req.question, retrieved, history)

    # 4) Call LLM (with messages including history + context)
    llm = get_llm_client()
    messages = [
        {"role": "system", "content": "You answer questions grounded in provided context."},
        {"role": "user", "content": prompt},
    ]
    answer = await llm.generate(messages)

    # 5) Update history
    STORE.append_message(req.session_id, "user", req.question)
    STORE.append_message(req.session_id, "assistant", answer)

    # 6) Return citations (chunk_id + score)
    citations = [Citation(chunk_id=cid, score=score) for (cid, score, _txt) in retrieved]
    turn_count = len(STORE.get_history(req.session_id)) // 2
    return QAResponse(answer=answer, citations=citations, turn_count=turn_count)

@app.post("/agent", response_model=AgentResponse)
async def agent(req: AgentRequest):
    steps = []
    query = req.query

    llm = get_llm_client()

    # --- Step 1: detect math ---
    if any(op in query for op in ["*", "+", "-", "/"]):
        expr = query.split("and")[0]
        expr = expr.lower().replace("what is", "").strip()
        expr = re.sub(r"[^0-9+\-*/(). ]", "", expr)
        try:
            result = eval_expr(expr)
        except Exception:
            result = "error"

        steps.append(Step(
            action="calculator",
            input=expr,
            output=result
        ))
    else:
        result = None

    # --- Step 2: knowledge ---
    query_lower = query.lower()

    needs_knowledge = any(word in query_lower for word in [
        "recursion", "explain", "define"
    ])

    if needs_knowledge:
        retrieved = await search_knowledge(STORE, req.query, k=4)

        context = "\n".join([chunk[2] for chunk in retrieved])

        steps.append(Step(
            action="knowledge_base",
            input=query,
            output=context[:200]
        ))
    else:
        context = ""

    # --- Step 3: final answer ---
    prompt = f"""
Use the following information to answer:

Context:
{context}

Math Result:
{result}

Question:
{query}
"""

    answer = await llm.generate([
        {"role": "user", "content": prompt}
    ])

    return AgentResponse(
        answer=answer,
        steps=steps
    )