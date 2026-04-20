from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_added: int


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    embedding: list[float]


class QARequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    score: float

class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    turn_count: int

class AgentRequest(BaseModel):
    session_id: str
    query: str

class Step(BaseModel):
    action: str
    input: str
    output: str

class AgentResponse(BaseModel):
    answer: str
    steps: list[Step]