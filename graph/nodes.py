from datetime import date
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END

from core.types import State, CustomerAction
from services.llm import llm
from services.rag import Build_Rag_index
from services.stocks import StockDataFetcher

# Initialize stock fetcher with dummy ticker
fetcher = StockDataFetcher(ticker="")


def know_base_1(state: State) -> dict:
    """Node wrapper for knowledge base search."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        
        # Safe argument retrieval
        if isinstance(tool_call, dict):
            args = tool_call.get('args', {})
            tool_call_id = tool_call.get("id")
        else:
            # Handle object-like tool call if necessary
            args = getattr(tool_call, 'args', {})
            tool_call_id = getattr(tool_call, 'id', None)
        
        query = args.get("search_query", state.get("query", ""))
        
        doc_path = state.get("doc_path")
        
        from core.types import RagIndexConfig
        
        config = RagIndexConfig(
            user_id="default_user", # Placeholder, ideally from state
            query=query,
            data_source="filepath", # Defaulting to filepath as per likely usage
            doc_path=doc_path or "uploaded_documents" # Default path
        )
        
        from services.rag import RagIndexOrchestrator
        orchestrator = RagIndexOrchestrator(config)
        result = orchestrator.create_knowledge_base()

        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call_id
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error accessing knowledge base: {str(e)}",
            tool_call_id=tool_call["id"] if 'tool_call' in locals() else "unknown"
        )]}


def company_information(state: State) -> dict:
    """Node wrapper for company information."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        result = fetcher._company_information()
        
        temp_fetcher = StockDataFetcher(ticker)
        result = temp_fetcher._company_information()
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error fetching company info: {str(e)}",
            tool_call_id=tool_call["id"]
        )]}

def last_dividend_and_earnings_date(state: State) -> dict:
    """Node wrapper for dividend/earnings dates."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        temp_fetcher = StockDataFetcher(ticker)
        result = temp_fetcher._last_dividend_and_earnings_date()
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error fetching dividend/earnings info: {str(e)}",
            tool_call_id=tool_call["id"]
        )]}

def summary_of_mutual_fund_holders(state: State) -> dict:
    """Node wrapper for mutual fund holders."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        temp_fetcher = StockDataFetcher(ticker)
        result = temp_fetcher._summary_of_mutual_fund_holders()
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error fetching mutual fund holders: {str(e)}",
            tool_call_id=tool_call["id"]
        )]}

def summary_of_institutional_holders(state: State) -> dict:
    """Node wrapper for institutional holders."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        temp_fetcher = StockDataFetcher(ticker)
        result = temp_fetcher._summary_of_institutional_holders()
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error fetching institutional holders: {str(e)}",
            tool_call_id=tool_call["id"]
        )]}

def stock_grade_upgrades_downgrades(state: State) -> dict: 
    """Node wrapper for stock grades."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        temp_fetcher = StockDataFetcher(ticker)
        result = temp_fetcher._stock_grade_upgrades_downgrades()
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error fetching stock grades: {str(e)}",
            tool_call_id=tool_call["id"]
        )]}

def stock_splits_history(state: State) -> dict:
    """Node wrapper for stock splits."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        temp_fetcher = StockDataFetcher(ticker)
        result = temp_fetcher._stock_splits_history()
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error fetching stock splits: {str(e)}",
            tool_call_id=tool_call["id"]
        )]}

def stock_news(state: State) -> dict:
    """Node wrapper for stock news."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        temp_fetcher = StockDataFetcher(ticker)
        result = temp_fetcher._stock_news()
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error fetching stock news: {str(e)}",
            tool_call_id=tool_call["id"]
        )]}

def stock_info(state: State) -> dict:
    """Node wrapper for stock price info."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        temp_fetcher = StockDataFetcher(ticker)
        result = temp_fetcher._stock_info()
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error fetching stock price info: {str(e)}",
            tool_call_id=tool_call["id"]
        )]}

def stock_history(state: State) -> dict:
    """Node wrapper for stock historical data."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        temp_fetcher = StockDataFetcher(ticker)
        result = temp_fetcher._stock_history(period="1mo")
        
        import json
        tool_message = ToolMessage(
            content=json.dumps(result),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        print(f"DEBUG: stock_history error: {e}")
        return {"messages": [ToolMessage(
            content=f"Error fetching stock history: {str(e)}",
            tool_call_id=tool_call["id"] if 'tool_call' in locals() else "unknown"
        )]}


def stock_sentiment(state: State) -> dict:
    """Node wrapper for stock sentiment analysis based on news."""
    try:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        
        ticker = args.get("ticker", "").upper()
        temp_fetcher = StockDataFetcher(ticker)
        news = temp_fetcher._stock_news()
        
        if not news:
            return {"messages": [ToolMessage(
                content=str({"sentiment_score": 0, "error": "No news found"}),
                tool_call_id=tool_call["id"]
            )]}
        
        # Construct prompt for sentiment
        news_text = "\n".join([f"- {n.get('title')}: {n.get('summary', '')}" for n in news[:5]])
        sentiment_prompt = f"""
        Analyze the following news for {ticker} and provide a sentiment score between -1 and 1.
        -1 is extremely bearish, 0 is neutral, and 1 is extremely bullish.
        Return ONLY a JSON object with the keys 'score' (float) and 'label' (string: Bullish, Bearish, or Neutral).
        
        News:
        {news_text}
        """
        
        from services.llm import llm
        sentiment_response = llm.invoke([
            SystemMessage(content="You are a financial sentiment analyzer. Output only JSON."), 
            HumanMessage(content=sentiment_prompt)
        ])
        
        import json
        import re
        try:
            # Extract JSON from potential markdown
            match = re.search(r'\{.*\}', sentiment_response.content, re.DOTALL)
            if match:
                sentiment_data = json.loads(match.group())
                score = sentiment_data.get("score", 0)
                label = sentiment_data.get("label", "Neutral")
            else:
                score = 0
                label = "Neutral"
        except:
            score = 0
            label = "Neutral"
            
        import json
        tool_message = ToolMessage(
            content=json.dumps({"sentiment_score": score, "sentiment_label": label}),
            tool_call_id=tool_call["id"]
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        return {"messages": [ToolMessage(
            content=f"Error performing sentiment analysis: {str(e)}",
            tool_call_id=tool_call["id"] if 'tool_call' in locals() else "unknown"
        )]}


# ============= SYSTEM PROMPT =============
prompt = """
You are a financial research assistant, always be formal to users.
You have access to multiple tools: Yahoo Finance tools for stock data and a Knowledge Base tool for platform-specific questions. 

Your job is to:
1. When you receive a user query, decide which tool to call (if any) to get the information
2. After receiving tool results, synthesize them into a clear, helpful answer for the user
3. DO NOT call tools multiple times - call once, then provide your final answer

Guidelines:
- If the query is about platform features, investing basics, withdrawals, deposits, account, or general FAQs → use the knowledge_base tool
- If the query is about stock tickers, prices, dividends, earnings, institutional holders, upgrades/downgrades, splits, or news → choose the most relevant Yahoo Finance tool
- If the user asks for a chart, a trend, or how a stock has performed recently (e.g., "See chart for NVDA", "AAPL trend", "MSFT performance") → you MUST use the get_stock_trend tool.
- If the user asks for the market sentiment, the "vibe", "sentiment analysis", or if a stock is bullish/bearish based on current news → you MUST use the get_stock_sentiment tool.
- After getting tool results, provide a formal and clear summary of the findings. If you used a chart or sentiment tool, specifically mention that you have provided the visual data in the response.
- Do not invent information - only use what the tools provide
- If information cannot be found, say: "I don't have that information in my current data sources"
"""

# ============= COMPLIANCE CHECK =============
def compliance_check(query: str) -> dict:
    """Compliance filter - checks if query is investment-related."""
    compliance_prompt = f"""
    You are a compliance filter for a financial research assistant.
    
    User query: "{query}"
    
    Respond with ONLY one word:
    - "APPROVED" if the query is about investments, stocks, the platform, or financial topics
    - "REJECTED" if the query is unrelated to finance/investments
    - "MALICIOUS" if the query contains malicious and curse words
    
    Response:"""
    
    try:
        response = llm.invoke(compliance_prompt)
        decision = response.content.strip().upper()
        
        if decision == "APPROVED":
            return {"approved": True}
        
        elif decision == "MALICIOUS":
            rejection_msg = (
                "Malicious words not allowed. Adjust your prompt."
            )
            return {"approved": False, "message": rejection_msg}
        else:
            rejection_msg = (
                "I'm a financial research assistant focused on investment topics and the platform's knowledge base. "
                "I can help with stock information, company data, and platform-related questions."
            )
            return {"approved": False, "message": rejection_msg}
            
    except Exception as e:
        return {
            "approved": False, 
            "message": "I'm experiencing technical difficulties. Please try again."
        }


def chatbot(state: State) -> dict:
    """Main chatbot node with compliance check and LLM call."""
    try:
        query = state.get("query", "")
        messages = state.get("messages", [])
        
        # Check if we just got tool results back
        has_tool_message = any(isinstance(m, ToolMessage) for m in messages)
        
        # Only run compliance check on initial query
        if not messages:
            compliance_result = compliance_check(query)
            
            if not compliance_result["approved"]:
                return {
                    "messages": [AIMessage(content=compliance_result["message"])]
                }
        
        # Build messages
        full_messages = [SystemMessage(content=prompt)]
        
        # Add conversation history
        full_messages.extend(messages)
        
        # Add current query only if this is the first call
        if not messages:
            full_messages.append(HumanMessage(content=query))
        
        # If we have tool results, don't bind tools (just respond)
        # If we don't have tool results, bind tools (to make a tool call)
        if has_tool_message:
            # Just respond with LLM, no tools
            response = llm.invoke(full_messages)
        else:
            # Bind tools for initial call
            llm_with_tools = llm.bind_tools([CustomerAction])
            response = llm_with_tools.invoke(full_messages)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"DEBUG: AI initiated tool calls: {response.tool_calls}")
        
        return {"messages": [response]}
        
    except Exception as e:
        print("chatbot_error:", e)
        return {
            "messages": [
                AIMessage(content="I'm experiencing technical difficulties. Please try again.")
            ]
        }


def route_message(state: State) -> str:
    """Route to the appropriate action node based on tool call."""
    
    last_message = state["messages"][-1]

    if not (isinstance(last_message, AIMessage) and last_message.tool_calls):
        return END
    
    tool_call = last_message.tool_calls[0]
    args = tool_call.get('args', {})

    if args.get("get_company_info"):
        return "company_information"
    if args.get("get_dividend_earnings"):
        return "last_dividend_and_earnings_date"
    if args.get("search_knowledge_base"):
        return "know_base_1"
    if args.get("get_mutual_fund_holders"):
        return "summary_of_mutual_fund_holders"
    if args.get("get_institutional_holders"):
        return "summary_of_institutional_holders"
    if args.get("get_stock_grades"):
        return "stock_grade_upgrades_downgrades"
    if args.get("get_stock_splits"):
        return "stock_splits_history"
    if args.get("get_stock_news"):
        return "stock_news"
    if args.get("get_stock_price"):
        return "stock_info"
    if args.get("get_stock_trend"):
        return "stock_history"
    if args.get("get_stock_sentiment"):
        return "stock_sentiment"
    
    return END
