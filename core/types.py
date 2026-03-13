import os
from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, validator

class State(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    tool_output: Optional[str]
    doc_path: Optional[str]


class RagIndexConfig(BaseModel):
    user_id: str
    query: str = ""
    ticker: Optional[str] = None
    data_source: Literal["filepath", "database", "s3"]
    doc_path: Optional[str] = None
    s3_prefix: Optional[str] = None
    
    @validator('doc_path')
    def validate_filepath(cls, v, values):
        """Validate filepath exists when data_source is filepath"""
        if values.get('data_source') == 'filepath':
            if not v:
                raise ValueError('doc_path is required when data_source is filepath')
            if not os.path.exists(v):
                raise ValueError(f'doc_path does not exist: {v}')
        return v
    
    @validator('s3_prefix')
    def validate_s3(cls, v, values):
        """Validate s3_prefix when data_source is s3"""
        if values.get('data_source') == 's3' and not v:
            raise ValueError('s3_prefix is required when data_source is s3')
        return v


class CustomerAction(BaseModel):
    """Handle customer financial research requests."""
    
    get_company_info: bool = Field(default=False, description="True if user wants company information")
    get_dividend_earnings: bool = Field(default=False, description="True if user wants dividend dates")
    search_knowledge_base: bool = Field(default=False, description="True if user has questions about the platform knowledge base")
    get_mutual_fund_holders: bool = Field(default=False, description="True if user wants mutual fund holders")
    get_institutional_holders: bool = Field(default=False, description="True if user wants institutional holders")
    get_stock_grades: bool = Field(default=False, description="True if user wants stock ratings")
    get_stock_splits: bool = Field(default=False, description="True if user wants stock splits")
    get_stock_news: bool = Field(default=False, description="True if user wants stock news")
    get_stock_price: bool = Field(default=False, description="True if user wants stock price")
    get_stock_trend: bool = Field(default=False, description="True if user wants to see a price chart or historical performance")
    get_stock_sentiment: bool = Field(default=False, description="True if user wants an AI sentiment analysis of the stock news")
    ticker: Optional[str] = Field(default=None, description="Stock ticker symbol")
    search_query: Optional[str] = Field(default=None, description="Search query for knowledge base")
