from fastapi import FastAPI, File, UploadFile, HTTPException
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

from graph.workflow import app as graph_app
from core.types import State
from langchain_core.messages import ToolMessage, AIMessage

# Initialize FastAPI
app = FastAPI(
    title="Financial Research Assistant API",
    description="API for financial research and Investment platform queries",
    version="1.0.0"
)

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

# Store the current document path globally
current_doc_path = None


# ============= REQUEST/RESPONSE MODELS =============
class QueryRequest(BaseModel):
    query: str
    


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
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to be used in the knowledge base.
    
    - **file**: PDF file to upload
    """
    global current_doc_path
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Create unique filename
        filepath = UPLOAD_DIR / file.filename
        
        # Save the file
        with filepath.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update the global document path
        current_doc_path = str(filepath.absolute())
        
        return {
            "filename": file.filename,
            "filepath": str(filepath),
            "message": f"Document '{file.filename}' uploaded successfully and will be used for knowledge base queries",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )
    finally:
        file.file.close()


@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    try:
        latest_doc_path = get_latest_uploaded_document()
        
        # Create a unique thread_id for this request to ensure fresh context
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "query": request.query,
            "messages": [],
            "tool_output": None,
            "doc_path": latest_doc_path 
        }
        
        print(f"DEBUG: Processing query with thread_id: {thread_id}")
        final_state = graph_app.invoke(initial_state, config=config)
        result = extract_final_response(final_state)
        
        return {
            "query": request.query,
            "response": result["response"],
            "tool_used": result["tool_used"],
            "chart_data": result.get("chart_data"),
            "sentiment_score": result.get("sentiment_score"),
            "sentiment_label": result.get("sentiment_label"),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def get_latest_uploaded_document() -> Optional[str]:
    """Get the most recently uploaded PDF (survives restarts)"""
    if not UPLOAD_DIR.exists():
        return None
    
    pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        return None
    
    # Return newest PDF by modification time
    latest_pdf = max(pdf_files, key=os.path.getmtime)
    return str(latest_pdf.absolute())



@app.delete("/document")
async def delete_uploaded_document():
    """
    Delete the currently uploaded document.
    """
    global current_doc_path
    
    # If global path is empty, try to get the latest from disk
    doc_to_delete = current_doc_path or get_latest_uploaded_document()
    
    if not doc_to_delete:
        raise HTTPException(
            status_code=404,
            detail="No document currently uploaded"
        )
    
    try:
        if os.path.exists(doc_to_delete):
            os.remove(doc_to_delete)
            filename = os.path.basename(doc_to_delete)
            current_doc_path = None
            
            return {
                "message": f"Document '{filename}' deleted successfully",
                "status": "success"
            }
        else:
            current_doc_path = None
            raise HTTPException(
                status_code=404,
                detail="Document file not found"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

@app.get("/document/status")
async def get_document_status():
    latest_doc = get_latest_uploaded_document()
    if latest_doc and os.path.exists(latest_doc):
        return {
            "status": "success",
            "has_document": True,
            "filename": os.path.basename(latest_doc),
            "filepath": latest_doc,
            "auto_detected": True  # Survives restarts!
        }
    return {
        "status": "success",
        "has_document": False,
        "message": "No PDF documents found"
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
        port=8000,
        reload=True
    )