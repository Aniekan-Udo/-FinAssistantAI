import os
import hashlib
from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging

from core.config import logger
from core.database import Base, IndexVersion
from core.types import RagIndexConfig
from services.document import DocumentLoader
from services.vector import (
    EmbeddingManager, 
    VectorStoreFactory, 
    IndexBuilder, 
    QueryExecutor, 
    _vectorstore_cache, 
    _vectorstore_lock
)


class VersionTracker:
    """Responsible for tracking and managing index versions"""
    
    def __init__(self, db_uri: str):
        engine = create_engine(db_uri)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)
    
    def create_version(
        self, 
        user_id: str, 
        documents: List,
        auto_activate: bool = True
    ) -> tuple[str, bool]:
        """Create version with deduplication. Returns: (version, is_new)"""
        session = self.Session()
        try:
            content_hash = self._hash(documents)
            
            # Check if content exists
            existing = session.query(IndexVersion)\
                .filter_by(user_id=user_id, content_hash=content_hash)\
                .first()
            
            if existing:
                if auto_activate:
                    self.activate(user_id, existing.version)
                return existing.version, False
            
            # Create new version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"{timestamp}_{content_hash}"
            
            record = IndexVersion(
                user_id=user_id,
                version=version,
                content_hash=content_hash,
                doc_count=len(documents),
                is_active=auto_activate
            )
            
            if auto_activate:
                session.query(IndexVersion)\
                    .filter_by(user_id=user_id)\
                    .update({"is_active": False})
            
            session.add(record)
            session.commit()
            
            return version, True
        finally:
            session.close()
    
    def get_active(self, user_id: str) -> Optional[IndexVersion]:
        """Get currently active version"""
        session = self.Session()
        try:
            return session.query(IndexVersion)\
                .filter_by(user_id=user_id, is_active=True)\
                .first()
        finally:
            session.close()
    
    def activate(self, user_id: str, version: str):
        """Switch to specific version"""
        session = self.Session()
        try:
            session.query(IndexVersion)\
                .filter_by(user_id=user_id)\
                .update({"is_active": False})
            
            session.query(IndexVersion)\
                .filter_by(user_id=user_id, version=version)\
                .update({"is_active": True})
            
            session.commit()
        finally:
            session.close()
    
    def rollback(self, user_id: str) -> str:
        """Rollback to previous version"""
        session = self.Session()
        try:
            versions = session.query(IndexVersion)\
                .filter_by(user_id=user_id)\
                .order_by(IndexVersion.created_at.desc())\
                .limit(2)\
                .all()
            
            if len(versions) < 2:
                raise ValueError("No previous version to rollback to")
            
            previous = versions[1]
            self.activate(user_id, previous.version)
            return previous.version
        finally:
            session.close()
    
    def cleanup(self, user_id: str, keep_last_n: int = 5):
        """Keep only the last N versions"""
        session = self.Session()
        try:
            versions = session.query(IndexVersion)\
                .filter_by(user_id=user_id)\
                .order_by(IndexVersion.created_at.desc())\
                .all()
            
            deleted_count = 0
            for version in versions[keep_last_n:]:
                if not version.is_active:
                    session.delete(version)
                    deleted_count += 1
            
            session.commit()
            return deleted_count
        finally:
            session.close()
    
    def get_table_name(self, user_id: str, version: str) -> str:
        """Get table name for version"""
        return f"rag_{user_id}_{version}"
    
    def list_versions(self, user_id: str, limit: int = 10) -> List[IndexVersion]:
        """List recent versions"""
        session = self.Session()
        try:
            return session.query(IndexVersion)\
                .filter_by(user_id=user_id)\
                .order_by(IndexVersion.created_at.desc())\
                .limit(limit)\
                .all()
        finally:
            session.close()
    
    def _hash(self, documents: List) -> str:
        """Generate 8-char hash"""
        content = "".join([getattr(doc, 'text', str(doc)) for doc in documents])
        return hashlib.sha256(content.encode()).hexdigest()[:8]


# Helper for lazy instantiation
_VERSION_TRACKER = None

def get_version_tracker() -> Optional[VersionTracker]:
    global _VERSION_TRACKER
    if _VERSION_TRACKER:
        return _VERSION_TRACKER
    
    uri = os.getenv("POSTGRES_ASYNC_URI")
    if not uri:
        logger.warning("POSTGRES_ASYNC_URI not set - VersionTracker disabled")
        return None
        
    try:
        _VERSION_TRACKER = VersionTracker(uri)
        return _VERSION_TRACKER
    except Exception as e:
        logger.error(f"Failed to initialize VersionTracker: {e}")
        return None


class RagIndexOrchestrator:
    """
    Orchestrates the RAG index creation and querying process.
    Coordinates between different components without doing the work itself.
    """
    
    def __init__(
        self, 
        config: RagIndexConfig, 
        version_tracker: VersionTracker = None
    ):
        self.config = config
        self.version_tracker = version_tracker or get_version_tracker()
        self.version = None
        self.table_name = None
        
        # Initialize components
        self.document_loader = DocumentLoader(config)
        self.embedding_manager = EmbeddingManager()
        self.index_builder = IndexBuilder(self.embedding_manager)
        self.query_executor = QueryExecutor()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def create_knowledge_base(self) -> dict:
        """
        Main entry point for creating knowledge base and executing query.
        Orchestrates the entire workflow.
        """
        cache_key = f"{self.config.user_id}_{self.config.data_source}"
        
        # Check cache first
        if cache_key in _vectorstore_cache:
            logger.info(f"Using cached vector store for {cache_key}")
            index = _vectorstore_cache[cache_key]
            result = self.query_executor.execute_query(index, self.config.query)
            result["version"] = self.version
            result["table_name"] = self.table_name
            return result
        
        with _vectorstore_lock:
            # Double-check after acquiring lock
            if cache_key in _vectorstore_cache:
                logger.info(f"Using cached vector store (after lock) for {cache_key}")
                index = _vectorstore_cache[cache_key]
                result = self.query_executor.execute_query(index, self.config.query)
                result["version"] = self.version
                result["table_name"] = self.table_name
                return result
            
            logger.info(f"Cache miss for {cache_key}, building new index")
            
            # Step 1: Load documents
            documents = self.document_loader.load()
            
            if not documents:
                raise RuntimeError(
                    f"Failed to load documents from {self.config.data_source}"
                )
            
            # Step 2: Create/get version
            try:
                self.version, is_new = self.version_tracker.create_version(
                    user_id=self.config.user_id,
                    documents=documents,
                    auto_activate=True
                )
                
                self.table_name = self.version_tracker.get_table_name(
                    self.config.user_id, 
                    self.version
                )
                
                logger.info(
                    f"Version: {self.version} | New: {is_new} | Table: {self.table_name}"
                )
                
                if not is_new:
                    logger.info(f"Content unchanged - reusing version {self.version}")
                
            except Exception as e:
                logger.error(f"Version creation failed: {e}", exc_info=True)
                raise RuntimeError(f"Failed to create version: {e}")
            
            # Step 3: Create vector store
            vector_store = VectorStoreFactory.create_pgvector_store(self.table_name)
            
            # Step 4: Build index
            index = self.index_builder.build_index(documents, vector_store)
            
            # Step 5: Cache the index
            _vectorstore_cache[cache_key] = index
            logger.info(
                f"Index cached. Cache size: {len(_vectorstore_cache)}/{_vectorstore_cache.maxsize}"
            )
            
            # Step 6: Execute query
            result = self.query_executor.execute_query(index, self.config.query)
            result["version"] = self.version
            result["table_name"] = self.table_name
            
            return result


class Build_Rag_index:
    """
    Backwards compatibility wrapper.
    Delegates to RagIndexOrchestrator.
    """
    
    def __init__(self, config: RagIndexConfig, version_tracker: VersionTracker = None):
        self.orchestrator = RagIndexOrchestrator(config, version_tracker)
        self.config = config
        self.version = None
        self.table_name = None
    
    def create_kb(self):
        """Backwards compatible method"""
        result = self.orchestrator.create_knowledge_base()
        self.version = self.orchestrator.version
        self.table_name = self.orchestrator.table_name
        return result
