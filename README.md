# Agentic RAG with an Evaluation Harness (and an MCP server)

A handbook question-answering assistant built the way production RAG systems are built:
**hybrid retrieval, grounded answers with citations, abstention when evidence is missing, and automated
evaluation that fails CI when quality drops.** It also exposes its tools over **MCP**, so any MCP client can use it.

```
Question -> Rewrite -> Hybrid retrieve (BM25 + dense, fused with RRF) -> Grade evidence
                                                                          |-- weak  -> abstain
                                                                          '-- good  -> Answer + citations
                         Eval harness: hit@k, MRR, answer match, citation precision, faithfulness, abstention
```

## Why this project
Most RAG demos stop at "chat with your PDF". This one measures itself:
- **Hybrid search**: BM25 for exact terms plus dense vectors for meaning, merged with Reciprocal Rank Fusion.
- **Abstention**: a coverage check refuses to answer when the retrieved text doesn't support the question.
- **Citations**: every answer points to the handbook section it came from.
- **Eval gate**: `python -m evals.run_eval` scores a labelled question set and exits non-zero below thresholds. CI runs it on every push.
- **MCP server**: `search_handbook` and `ask_handbook` tools for Claude Desktop, IDEs or other agents.
- **Runs offline**: no API key needed (BM25 + TF-IDF + extractive answers). Add Azure OpenAI keys to switch to embeddings and LLM answers.

## Quick start
```bash
git clone https://github.com/Sp7906/agentic-rag-eval.git && cd agentic-rag-eval
cp .env.example .env            # optional: add Azure OpenAI credentials
docker compose up --build       # UI: http://localhost:5173   API docs: http://localhost:8000/docs
```
Without Docker:
```bash
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
cd frontend && npm install && npm run dev
```

## Evaluation
```bash
cd backend
python -m pytest -q                  # unit tests + quality gate
python -m evals.run_eval --offline   # prints metrics, writes evals/report.md
```
Offline results on the bundled 17-question set (14 answerable, 3 not in the handbook):

| Metric | Result | Gate |
|---|---|---|
| hit@3 (right document retrieved) | 1.00 | >= 0.85 |
| MRR | 1.00 | >= 0.75 |
| answer match | 1.00 | >= 0.80 |
| citation precision | 1.00 | >= 0.85 |
| abstain accuracy (unanswerable questions) | 1.00 | >= 0.90 |
| false abstain rate | 0.00 | <= 0.10 |

Caveat: the corpus is small and synthetic, and the offline synonym list was tuned on it, so these scores are a
regression baseline, not a general benchmark. Add your own documents to `docs/` and questions to
`backend/evals/dataset.jsonl`, then re-run.

## MCP server
```bash
cd backend && pip install -r requirements-mcp.txt
python -m app.mcp_server
```
Claude Desktop config (use absolute paths):
```json
{ "mcpServers": { "northwind-handbook": {
    "command": "python", "args": ["-m", "app.mcp_server"], "cwd": "/absolute/path/to/backend" } } }
```

## Project layout
```
docs/                    sample handbook (markdown)
backend/app/ingest.py    section-level chunking
backend/app/retriever.py BM25 + dense + RRF + light rerank
backend/app/agent.py     rewrite -> retrieve -> grade -> answer / abstain
backend/app/mcp_server.py  MCP tools
backend/evals/           labelled dataset + metrics runner (CI quality gate)
backend/tests/           unit tests and eval gate test
frontend/                React UI with citations and agent trace
```

## Roadmap
Azure AI Search backend, cross-encoder reranker, LLM-as-judge faithfulness, query decomposition for multi-hop questions, tracing and cost dashboard.
