import os
from langchain_groq import ChatGroq
from core.config import ChatGroq_API_KEY

# Model ids are configurable because Groq retires models; check https://console.groq.com/docs/models
PRIMARY_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

# Primary model, with a smaller model as fallback when the free-tier rate limit is hit
_primary = ChatGroq(model=PRIMARY_MODEL, api_key=ChatGroq_API_KEY, max_tokens=1024, max_retries=1)
_fallback = ChatGroq(model=FALLBACK_MODEL, api_key=ChatGroq_API_KEY, max_tokens=1024, max_retries=1)


class _LLMWithFallback:
    """Keeps .invoke and .bind_tools working while falling back to the smaller model."""

    def invoke(self, *args, **kwargs):
        return _primary.with_fallbacks([_fallback]).invoke(*args, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return _primary.bind_tools(tools, **kwargs).with_fallbacks([_fallback.bind_tools(tools, **kwargs)])


llm = _LLMWithFallback()
