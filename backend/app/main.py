from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import agent

app = FastAPI(title="Agentic RAG with Evals")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(agent.get_retriever().chunks)}


@app.post("/ask")
def ask(body: Ask):
    return agent.answer(body.question, k=body.k)


@app.get("/search")
def search(q: str, k: int = 4):
    hits = agent.get_retriever().search(q, k)
    return [{"id": h.chunk.id, "source": h.chunk.source, "heading": h.chunk.heading,
             "score": h.score, "text": h.chunk.text} for h in hits]
