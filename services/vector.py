from typing import List
from threading import RLock
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex

from core.config import logger
from core.database import DatabaseURIParser

# Cache Embedding
_cache_embed = TTLCache(maxsize=5, ttl=7200)
_embed_lock = RLock()

# Cache Vector Store
_vectorstore_cache = TTLCache(maxsize=20, ttl=3600)
_vectorstore_lock = RLock()


class EmbeddingManager:
    """Responsible for managing embedding models with caching"""
    
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def get_embedding_model():
        """Get or create cached embedding model"""
        cache_key = "shared_embed"

        if cache_key in _cache_embed:
            return _cache_embed[cache_key]
        
        with _embed_lock:
            if cache_key in _cache_embed:
                return _cache_embed[cache_key]
            
            try:
                logger.info("Setting up embedding model")
                embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                _cache_embed[cache_key] = embed_model
                logger.info("Embedding model created and cached")
                return embed_model
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
                raise


class VectorStoreFactory:
    """Responsible for creating and configuring vector stores"""
    
    @staticmethod
    def create_pgvector_store(table_name: str) -> PGVectorStore:
        """Create PGVector store with configuration"""
        parsed = DatabaseURIParser.parse()
        
        return PGVectorStore.from_params(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path.lstrip('/'),
            user=parsed.username,
            password=parsed.password,
            table_name=table_name,
            embed_dim=384,
            hybrid_search=True,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops"
            }
        )


class IndexBuilder:
    """Responsible for building vector indexes from documents"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
    
    def build_index(
        self, 
        documents: List, 
        vector_store: PGVectorStore
    ) -> VectorStoreIndex:
        """Build vector index from documents"""
        from llama_index.core.node_parser import SimpleNodeParser
        
        logger.info("Building vector index")
        
        parser = SimpleNodeParser.from_defaults(chunk_size=256, chunk_overlap=100)
        nodes = parser.get_nodes_from_documents(documents)
        
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                nodes=nodes,
                embed_model=self.embedding_manager.get_embedding_model()
            )
            
            logger.info("Index created successfully")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}", exc_info=True)
            raise


class QueryExecutor:
    """Responsible for executing queries against the index"""
    
    @staticmethod
    def execute_query(index: VectorStoreIndex, query: str, top_k: int = 3) -> dict:
        """Execute retrieval query"""
        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        
        return {
            "content": " ".join([r.text for r in results]),
            "sources": [
                {
                    "page": r.metadata.get('page_label', 'N/A'),
                    "file": r.metadata.get('file_name', 'Unknown'),
                    "score": r.score
                }
                for r in results
            ]
        }
