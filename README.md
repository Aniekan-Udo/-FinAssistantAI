# AI Financial Research Assistant

A LangGraph agent that answers investment questions by routing each query to the right source: **live Yahoo Finance market data** (11 tools) or a **pgvector knowledge base** built from an investment platform's documents. Visitors can upload their own PDF and ask questions about it.

**Live demo: https://finassistantai-hxvs.onrender.com** · Runs on Render's free tier, which sleeps when idle, so the first request can take ~50 s.

## Results

Measured with [`eval/run_eval.py`](eval/run_eval.py) against the running API (`openai/gpt-oss-120b` on Groq):

| Eval set | Questions | Tool routing accuracy | Knowledge-base answer accuracy | Latency p50 / p95 |
|---|---|---|---|---|
| Development set (used while fixing bugs) | 30 | 29/30 (97%) | 15/15 | 4.8 s / 9.3 s |
| Held-out set (never tuned against) | 10 | 10/10 | 6/6 | 5.3 s / 9.7 s |
| Held-out set on the **live deployment** | 10 | 10/10 | 6/6 | 2.7 s / 3.7 s |

The first two rows were run locally from Lagos and the last against the Render deployment, which sits closer to Groq. Latency is end to end: compliance check, tool call, data fetch and answer, about three LLM calls per question. These are small eval sets, so treat them as a sanity check, not a benchmark.

## How it works

```
question ─► compliance filter ─► router LLM (structured tool call)
                                   ├─► Knowledge base: pgvector retrieval over the platform docs or your upload
                                   └─► Yahoo Finance: price · company info · dividends & earnings · fund & institutional
                                       holders · analyst grades · splits · news · price chart · AI news sentiment
                                 ─► answer LLM (uses only the retrieved data)
```

- **Agent:** LangGraph `StateGraph`. The router binds a single `CustomerAction` schema, and conditional edges dispatch to one of 11 tool nodes.
- **Retrieval:** LlamaIndex + pgvector (Neon Postgres) with hybrid search. Embeddings come from `BAAI/bge-small-en-v1.5` running on ONNX via fastembed, so no torch and it fits a 512 MB instance.
- **Versioned indexes:** each document set is hashed (SHA-256). Unchanged content reuses its existing index instead of being re-embedded, earlier versions can be rolled back, and a failed build removes its version so the next request retries cleanly.
- **Per-visitor isolation:** each browser session gets its own upload folder and index, and nobody sees anyone else's documents.
- **Guardrails:** a compliance filter for off-topic or abusive queries, per-IP rate limits (10/min, 100/day), 5 MB PDF cap, 500-character query cap, and an automatic fallback to a smaller model when the main model is rate limited.
- **Resilience:** tenacity retries, a circuit breaker on database loading, and TTL caches for the embedder and indexes.

## Run locally

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
cp .env.example .env        # add your Groq key and a Postgres URI with pgvector
uv run uvicorn app:app --port 8000
# open http://localhost:8000
python eval/run_eval.py http://localhost:8000 holdout_set
```

## Deploy

`render.yaml` defines a free Render web service. In Render: **New → Blueprint →** pick this repo, then set `API_KEY` and `POSTGRES_ASYNC_URI`. The FastAPI app serves the frontend itself, so no separate frontend host is needed.

## What I'd do next

- Grow the eval sets and score answer faithfulness with an LLM judge, not just key-fact matching
- Stream answers token by token to cut perceived latency
- Re-ranking for the knowledge base and citations in answers
- Fix the remaining routing miss ("What does Tesla do?" goes to the knowledge base)
