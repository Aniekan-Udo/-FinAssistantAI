from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import shutil
from pathlib import Path
import uvicorn
import uuid
import json
import ast
import re
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from graph.workflow import app as graph_app
from core.types import State
from langchain_core.messages import ToolMessage, AIMessage

# Initialize FastAPI
app = FastAPI(
    title="Financial Research Assistant API",
    description="API for financial research and Investment platform queries",
    version="1.0.0"
)

# Rate limits protect the free-tier Groq quota on the public demo
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)
DEMO_KB_DIR = Path("knowledge_base")  # bundled sample platform docs, used until a visitor uploads their own
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_QUERY_CHARS = 500
SESSION_RE = re.compile(r"^[0-9a-f]{32}$")


def session_dir(session_id: Optional[str]) -> Optional[Path]:
    """Each browser session gets its own upload folder; invalid ids get none."""
    if not session_id or not SESSION_RE.match(session_id):
        return None
    return UPLOAD_DIR / session_id


# ============= REQUEST/RESPONSE MODELS =============
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    


class QueryResponse(BaseModel):
    query: str
    response: str
    tool_used: Optional[str] = None
    chart_data: Optional[dict] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    status: str


class UploadResponse(BaseModel):
    filename: str
    filepath: str
    message: str
    status: str


class HealthResponse(BaseModel):
    status: str
    message: str


# ============= HELPER FUNCTIONS =============
def extract_final_response(state: State) -> dict:
    """Extract the final response from the graph state."""
    messages = state.get("messages", [])
    
    if not messages:
        return {
            "response": "No response generated",
            "tool_used": None
        }
    
    # Get the last AI message
    final_message = messages[-1]
    response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    # Process messages in reverse order to get the MOST RECENT tool results and response
    tool_used = None
    chart_data = None
    sentiment_score = None
    sentiment_label = None
    
    # Iterate in reverse to find the latest tools and summary
    for msg in reversed(messages):
        # 1. Look for the final AI response (non-tool-calling)
        if hasattr(msg, 'content') and msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
            if not response_text or response_text == str(msg):
                 response_text = msg.content

        # 2. Look for the most recent Tool results
        if isinstance(msg, ToolMessage) and not chart_data and sentiment_score is None:
            print(f"DEBUG: Found ToolMessage content: {msg.content[:100]}...")
            try:
                import json
                content_dict = json.loads(msg.content)
                if isinstance(content_dict, dict):
                    if "prices" in content_dict:
                        chart_data = content_dict
                        print("DEBUG: Extracted Chart Data")
                    if "sentiment_score" in content_dict:
                        sentiment_score = content_dict["sentiment_score"]
                        sentiment_label = content_dict.get("sentiment_label")
                        print(f"DEBUG: Extracted Sentiment: {sentiment_label} ({sentiment_score})")
            except Exception as e:
                print(f"DEBUG: Failed to parse ToolMessage JSON: {e}")

        # 3. Look for the AI message that initiated the tool call
        if hasattr(msg, 'tool_calls') and msg.tool_calls and not tool_used:
            tool_call = msg.tool_calls[0]
            args = tool_call.get('args', {})
            print(f"DEBUG: AI called tool with args: {args}")
            
            # Map the tool flags to readable names
            if args.get("search_knowledge_base"):
                tool_used = "Platform Knowledge Base"
            elif args.get("get_company_info"):
                tool_used = "Company Information"
            elif args.get("get_dividend_earnings"):
                tool_used = "Dividend & Earnings"
            elif args.get("get_mutual_fund_holders"):
                tool_used = "Mutual Fund Holders"
            elif args.get("get_institutional_holders"):
                tool_used = "Institutional Holders"
            elif args.get("get_stock_grades"):
                tool_used = "Stock Grades"
            elif args.get("get_stock_splits"):
                tool_used = "Stock Splits"
            elif args.get("get_stock_news"):
                tool_used = "Stock News"
            elif args.get("get_stock_price"):
                tool_used = "Stock Price"
            elif args.get("get_stock_trend"):
                tool_used = "Price History Chart"
            elif args.get("get_stock_sentiment"):
                tool_used = "AI Sentiment Analysis"

    return {
        "response": response_text or "No response generated.",
        "tool_used": tool_used,
        "chart_data": chart_data,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label
    }


# ============= API ENDPOINTS =============
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return {
        "status": "success",
        "message": "All systems operational"
    }


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("5/hour")
async def upload_document(request: Request, file: UploadFile = File(...), session_id: str = Form(...)):
    """
    Upload a PDF to this session's knowledge base (replaces any previous upload).

    - **file**: PDF file to upload (max 5 MB)
    - **session_id**: 32-char hex id generated by the browser
    """
    folder = session_dir(session_id)
    if folder is None:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="PDF must be 5 MB or smaller")

        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        filepath = folder / Path(file.filename).name
        filepath.write_bytes(content)

        return {
            "filename": filepath.name,
            "filepath": str(filepath),
            "message": f"Document '{filepath.name}' uploaded successfully and will be used for knowledge base queries",
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    finally:
        await file.close()


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute;100/day")
async def query_system(request: Request, body: QueryRequest):
    if len(body.query) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=400, detail=f"Query must be {MAX_QUERY_CHARS} characters or fewer")
    try:
        uploaded = get_latest_uploaded_document(body.session_id)
        kb_dir = str(Path(uploaded).parent) if uploaded else str(DEMO_KB_DIR.absolute())
        kb_owner = body.session_id if uploaded else "demo"
        
        # Create a unique thread_id for this request to ensure fresh context
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "query": body.query,
            "messages": [],
            "tool_output": None,
            "doc_path": kb_dir,
            "kb_owner": kb_owner,
        }
        
        print(f"DEBUG: Processing query with thread_id: {thread_id}")
        final_state = graph_app.invoke(initial_state, config=config)
        result = extract_final_response(final_state)
        
        return {
            "query": body.query,
            "response": result["response"],
            "tool_used": result["tool_used"],
            "chart_data": result.get("chart_data"),
            "sentiment_score": result.get("sentiment_score"),
            "sentiment_label": result.get("sentiment_label"),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def get_latest_uploaded_document(session_id: Optional[str]) -> Optional[str]:
    """Get the PDF uploaded in this session, if any"""
    folder = session_dir(session_id)
    if folder is None or not folder.exists():
        return None

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        return None
    return str(max(pdf_files, key=os.path.getmtime).absolute())


@app.delete("/document")
async def delete_uploaded_document(session_id: str):
    """Delete this session's uploaded document (falls back to the demo knowledge base)."""
    latest = get_latest_uploaded_document(session_id)
    if not latest:
        raise HTTPException(status_code=404, detail="No document currently uploaded")

    shutil.rmtree(session_dir(session_id), ignore_errors=True)
    return {
        "message": f"Document '{os.path.basename(latest)}' deleted successfully",
        "status": "success"
    }


@app.get("/document/status")
async def get_document_status(session_id: Optional[str] = None):
    latest_doc = get_latest_uploaded_document(session_id)
    if latest_doc:
        return {
            "status": "success",
            "has_document": True,
            "filename": os.path.basename(latest_doc),
            "auto_detected": True
        }
    return {
        "status": "success",
        "has_document": False,
        "message": "Using the demo investment-platform knowledge base"
    }


# ============= EXAMPLE QUERIES ENDPOINT =============
@app.get("/examples")
async def get_example_queries():
    """
    Get example queries for different use cases.
    """
    return {
        "platform_general": [
            "How do I create an account?",
            "How can I withdraw money?",
            "What investment options are available?",
            "How do I deposit money?"
        ],
        "stock_information": [
            "What is the current price of AAPL?",
            "Tell me about Microsoft's company information",
            "What are the institutional holders of TSLA?",
            "Show me recent news about Amazon"
        ],
        "stock_analysis": [
            "When is the next dividend date for AAPL?",
            "What are the recent upgrades and downgrades for GOOGL?",
            "Show me the stock split history of TSLA",
            "Who are the top mutual fund holders of MSFT?"
        ]
    }


# Mount static files at the root (must be after other routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# ============= RUN THE APPLICATION =============
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )