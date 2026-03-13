from langgraph.graph import StateGraph, START, END
from core.types import State
from graph.nodes import (
    chatbot,
    know_base_1,
    company_information,
    last_dividend_and_earnings_date,
    summary_of_mutual_fund_holders,
    summary_of_institutional_holders,
    stock_grade_upgrades_downgrades,
    stock_splits_history,
    stock_news,
    stock_info,
    stock_history,
    stock_sentiment,
    route_message
)

# ============= GRAPH CONSTRUCTION =============
graph = StateGraph(State)

# Add all nodes
graph.add_node("chatbot", chatbot)
graph.add_node("know_base_1", know_base_1)
graph.add_node("company_information", company_information)
graph.add_node("last_dividend_and_earnings_date", last_dividend_and_earnings_date)
graph.add_node("summary_of_mutual_fund_holders", summary_of_mutual_fund_holders)
graph.add_node("summary_of_institutional_holders", summary_of_institutional_holders)
graph.add_node("stock_grade_upgrades_downgrades", stock_grade_upgrades_downgrades) 
graph.add_node("stock_splits_history", stock_splits_history)
graph.add_node("stock_news", stock_news)
graph.add_node("stock_info", stock_info)
graph.add_node("stock_history", stock_history)
graph.add_node("stock_sentiment", stock_sentiment)

# Set entry point
graph.add_edge(START, "chatbot")

# Add conditional routing from chatbot to tools
graph.add_conditional_edges(
    "chatbot",
    route_message,
    {
        "know_base_1": "know_base_1",
        "company_information": "company_information",
        "last_dividend_and_earnings_date": "last_dividend_and_earnings_date",
        "summary_of_mutual_fund_holders": "summary_of_mutual_fund_holders",
        "summary_of_institutional_holders": "summary_of_institutional_holders",
        "stock_grade_upgrades_downgrades": "stock_grade_upgrades_downgrades",
        "stock_splits_history": "stock_splits_history",
        "stock_news": "stock_news",
        "stock_info": "stock_info",
        "stock_history": "stock_history",
        "stock_sentiment": "stock_sentiment",
        END: END
    }
)

# All tools return to chatbot for final synthesis
for node in ["know_base_1", "company_information", "last_dividend_and_earnings_date",
             "summary_of_mutual_fund_holders", "summary_of_institutional_holders",
             "stock_grade_upgrades_downgrades", "stock_splits_history", 
             "stock_news", "stock_info", "stock_history", "stock_sentiment"]:
    graph.add_edge(node, "chatbot")

# Compile the graph
app = graph.compile()
