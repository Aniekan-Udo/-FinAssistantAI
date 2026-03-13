from langchain_groq import ChatGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from core.config import ChatGroq_API_KEY

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=ChatGroq_API_KEY)

# Initialize embeddings
embedding = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
