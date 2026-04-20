from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, TypedDict

from .models import ChunkRecord

class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str

@dataclass
class InMemoryStore:
    # chunk_id -> chunk record
    chunks: Dict[str, ChunkRecord] = field(default_factory=dict)
    # doc_id -> list of chunk_ids
    doc_chunks: Dict[str, List[str]] = field(default_factory=dict)

    # session_id -> list of messages
    sessions: Dict[str, List[ChatMessage]] = field(default_factory=dict)

    def upsert_document_chunks(self, doc_id: str, new_chunks: list[ChunkRecord]) -> int:

        old_ids = self.doc_chunks.get(doc_id, [])
        for cid in old_ids:
            self.chunks.pop(cid, None)

        # insert new
        self.doc_chunks[doc_id] = []
        for ch in new_chunks:
            self.chunks[ch.chunk_id] = ch
            self.doc_chunks[doc_id].append(ch.chunk_id)

        return len(new_chunks)

    def count_chunks_for_doc(self, doc_id: str) -> int:
        return len(self.doc_chunks.get(doc_id, []))
    
    def get_history(self, session_id: str) -> List[ChatMessage]:
        return self.sessions.get(session_id, [])

    def append_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})