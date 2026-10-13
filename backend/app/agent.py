"""Agentic RAG loop: rewrite -> hybrid retrieve -> grade evidence -> answer with citations (or abstain)."""
import re
import time
from functools import lru_cache

from . import config, llm
from .ingest import load_chunks
from .retriever import HybridRetriever
from .text import coverage, expand_query, expand_terms, tokens

ABSTAIN_TEXT = "I couldn't find this in the handbook, so I won't guess."

ANSWER_SYSTEM = """Answer the question using ONLY the numbered context. Cite sources like [1].
Be concise. If the context does not contain the answer, reply exactly: INSUFFICIENT_CONTEXT"""


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    chunks = load_chunks(config.DOCS_DIR)
    if llm.enabled():
        try:
            return HybridRetriever(chunks, embed_fn=llm.embed)
        except Exception:  # noqa: BLE001 - fall back to offline retrieval
            pass
    return HybridRetriever(chunks)


def _cite(n: int, hit) -> dict:
    c = hit.chunk
    return {"n": n, "id": c.id, "source": c.source, "title": c.title,
            "heading": c.heading, "snippet": c.text}


def _extractive(question: str, hits) -> tuple[str, list[int]]:
    q = set(tokens(question)) | set(expand_terms(question))
    cands = []
    for n, h in enumerate(hits[:3], 1):
        for j, sent in enumerate(re.split(r"(?<=[.!?])\s+", h.chunk.text)):
            ov = len(q & set(tokens(sent)))
            if ov:
                cands.append((ov - 0.1 * n, n, j, sent))
    if not cands:
        return "", []
    cands.sort(reverse=True)
    best = cands[0][0]
    top_chunk = cands[0][1]  # extra sentences must come from the same chunk as the best one
    picked = sorted([c for c in cands[:3] if c[1] == top_chunk and c[0] >= 0.5 * best][:2],
                    key=lambda c: (c[1], c[2]))
    return " ".join(f"{c[3]} [{c[1]}]" for c in picked), sorted({c[1] for c in picked})


def _llm_answer(question: str, hits) -> tuple[str, list[int]]:
    ctx = "\n\n".join(f"[{n}] ({h.chunk.title} / {h.chunk.heading}) {h.chunk.text}"
                      for n, h in enumerate(hits, 1))
    text = llm.chat([{"role": "system", "content": ANSWER_SYSTEM},
                     {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}])
    if "INSUFFICIENT_CONTEXT" in text:
        return "", []
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
    return text, cited or [1]


def answer(question: str, k: int = config.TOP_K) -> dict:
    t0 = time.perf_counter()
    retriever = get_retriever()
    use_llm = llm.enabled()
    trace, mode = [], "azure" if use_llm else "offline"

    query = expand_query(question)
    if use_llm:
        try:
            query = llm.chat([
                {"role": "system", "content": "Rewrite the question as a short keyword-rich search query "
                                              "for an employee handbook. Return only the query."},
                {"role": "user", "content": question}])
        except Exception as e:  # noqa: BLE001
            trace.append({"step": "rewrite_error", "error": str(e)}); use_llm, mode = False, "offline-fallback"
    trace.append({"step": "rewrite", "query": query})

    hits = retriever.search(query, k)
    trace.append({"step": "retrieve", "dense": retriever.dense_kind,
                  "hits": [{"id": h.chunk.id, "score": h.score} for h in hits]})

    cov = max((coverage(question, h.chunk.search_text) for h in hits), default=0.0)
    grounded = cov >= config.MIN_COVERAGE
    trace.append({"step": "grade", "coverage": round(cov, 2), "threshold": config.MIN_COVERAGE,
                  "decision": "answer" if grounded else "abstain"})

    text, cited = "", []
    if grounded:
        try:
            text, cited = (_llm_answer if use_llm else _extractive)(question, hits)
        except Exception as e:  # noqa: BLE001
            trace.append({"step": "answer_error", "error": str(e)})
            text, cited = _extractive(question, hits); mode = "offline-fallback"

    abstained = not text
    return {
        "question": question,
        "answer": ABSTAIN_TEXT if abstained else text,
        "abstained": abstained,
        "citations": [_cite(n, hits[n - 1]) for n in cited],
        "retrieved": [{"id": h.chunk.id, "source": h.chunk.source, "score": h.score} for h in hits],
        "trace": trace,
        "mode": mode,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
