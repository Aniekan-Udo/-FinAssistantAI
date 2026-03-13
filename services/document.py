import os
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from pybreaker import CircuitBreaker
from llama_index.core import Document

import logging
from core.config import logger
from core.types import RagIndexConfig
from core.database import DatabaseURIParser, BrandDocument

class DocumentLoader:
    """Responsible for loading documents from various sources"""

    def __init__(self, config: RagIndexConfig):
        self.config = config
        self.db_breaker = CircuitBreaker(fail_max=5, timeout_duration=60)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def load_from_filepath(self) -> List:
        """Load documents from filepath with validation"""
        from llama_index.core import SimpleDirectoryReader
        
        try:
            logger.info(f"Loading documents from {self.config.doc_path}")
            
            if not os.path.exists(self.config.doc_path):
                raise FileNotFoundError(f"Path does not exist: {self.config.doc_path}")
            
            if not os.access(self.config.doc_path, os.R_OK):
                raise PermissionError(f"No read permission for: {self.config.doc_path}")
            
            documents = SimpleDirectoryReader(
                self.config.doc_path,
                required_exts=[".pdf", ".txt", ".docx", ".md"],
                errors='ignore'
            ).load_data()
            
            if not documents:
                raise ValueError(f"No valid documents found in {self.config.doc_path}")
            
            logger.info(f"Loaded {len(documents)} documents from filepath")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load from filepath: {e}", exc_info=True)
            raise RuntimeError(f"Filepath loading failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def load_from_s3(self) -> List:
        """Load documents from S3 with validation"""
        from llama_index.readers.s3 import S3Reader
        from botocore.exceptions import ClientError, NoCredentialsError
        
        try:
            logger.info(f"Loading documents from S3: {self.config.s3_prefix}")
            
            reader = S3Reader(bucket=self.config.s3_prefix)
            documents = reader.load_data(prefix=self.config.s3_prefix)
            
            if not documents:
                raise ValueError(f"No documents found in S3 bucket: {self.config.s3_prefix}")
            
            logger.info(f"Loaded {len(documents)} documents from S3")
            return documents
            
        except NoCredentialsError as e:
            logger.error("AWS credentials not found")
            raise PermissionError("AWS credentials not configured") from e
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise FileNotFoundError(f"S3 bucket not found: {self.config.s3_prefix}") from e
            elif error_code == 'AccessDenied':
                raise PermissionError(f"Access denied to S3 bucket: {self.config.s3_prefix}") from e
            else:
                raise RuntimeError(f"S3 error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load from S3: {e}", exc_info=True)
            raise RuntimeError(f"S3 loading failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def load_from_database(self) -> List[Document]:
        """Load documents from database using ORM"""
        from sqlalchemy import create_engine
        from sqlalchemy.exc import OperationalError, DatabaseError
        from sqlalchemy.orm import sessionmaker
        
        try:
            logger.info("Loading documents from database using ORM")
            
            parsed = DatabaseURIParser.parse()
            connection_string = DatabaseURIParser.build_connection_string(parsed)
            
            engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={
                    "connect_timeout": 10,
                    "options": "-c statement_timeout=30000"
                }
            )
            
            # Test connection
            with engine.connect() as conn:
                logger.info("Database connection successful")
            
            Session = sessionmaker(bind=engine)
            session = Session()
            
            try:
                brand_docs = session.query(BrandDocument)\
                    .filter(BrandDocument.user_id == self.config.user_id)\
                    .all()
                
                if not brand_docs:
                    raise ValueError(f"No documents found for user {self.config.user_id}")
                
                documents = []
                for doc in brand_docs:
                    documents.append(
                        Document(
                            text=doc.file_content,
                            metadata={
                                'filename': doc.filename,
                                'user_id': doc.user_id,
                                'doc_id': doc.id
                            }
                        )
                    )
                
                logger.info(f"Loaded {len(documents)} documents from database")
                return documents
                
            finally:
                session.close()
            
        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            raise ConnectionError(f"Cannot connect to database: {e}") from e
        except DatabaseError as e:
            logger.error(f"Database query error: {e}")
            raise RuntimeError(f"Database query failed: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load from database: {e}", exc_info=True)
            raise RuntimeError(f"Database loading failed: {e}")
    
    def load(self) -> List:
        """Load documents based on data source"""
        if self.config.data_source == "filepath":
            return self.load_from_filepath()
        elif self.config.data_source == "s3":
            return self.load_from_s3()
        elif self.config.data_source == "database":
            return self.load_from_database()
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")
