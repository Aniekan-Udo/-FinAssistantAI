import os
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

Base = declarative_base()


class IndexVersion(Base):
    """Track index versions"""
    __tablename__ = 'index_versions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), index=True)
    version = Column(String(100))
    content_hash = Column(String(16))
    doc_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)


class BrandDocument(Base):
    """ORM model for Brand_Document table"""
    __tablename__ = 'Brand_Document'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), index=True, nullable=False)
    filename = Column(String(500))
    file_content = Column(Text)


class DatabaseURIParser:
    """Responsible for parsing and cleaning database URIs"""
    
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def parse() -> str:
        """Parse and clean Neon DB URI"""
        from urllib.parse import urlparse
        
        raw_uri = os.getenv("POSTGRES_ASYNC_URI", "").strip()
        
        if not raw_uri:
            raise ValueError("POSTGRES_ASYNC_URI environment variable not set")
        
        # Clean URI
        raw_uri = raw_uri.replace("&channel_binding=require", "")
        raw_uri = raw_uri.replace("channel_binding=require&", "")
        raw_uri = raw_uri.replace("?channel_binding=require", "")
        
        return urlparse(raw_uri)
    
    @staticmethod
    def build_connection_string(parsed_uri) -> str:
        """Build PostgreSQL connection string"""
        return (
            f"postgresql://{parsed_uri.username}:{parsed_uri.password}"
            f"@{parsed_uri.hostname}:{parsed_uri.port or 5432}{parsed_uri.path}"
        )
