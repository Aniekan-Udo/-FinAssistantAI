# AI Financial Research Assistant

A production-ready **LangGraph + FastAPI** RAG system combining **real-time stock data (Yahoo Finance)** with an **investment platform's knowledge base (PDF RAG)**. Supports company info, institutional holders, dividends, news, and platform-specific FAQs.

## Features

- **Hybrid RAG**: Yahoo Finance APIs + PDF knowledge base
- **Automatic Routing**: LLM decides stock data vs Platform FAQ
- **Persistent Vector DB**: Chroma embeddings survive restarts
- **File Upload API**: Upload platform docs, auto-detected on restart
- **Production FastAPI**: CORS, health checks, OpenAPI docs
- **Groq Llama 3.3**: Fast, powerful financial reasoning
- **BAAI/bge-small-en-v1.5**: For fast embedding

## 🛠️ Quick Start (uv) - 50MB vs 400MB venv

```bash
# 1. Navigate to project
cd "FINANCIAL RAG SYSTEM"

# 2. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create lightweight project
uv init
uv pip install -r requirements.txt  # Installs your packagespython-dotenv pydantic

# 4. Run API
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
