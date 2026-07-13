from app.config import settings
from app.llm.base import LLMProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.openai_provider import OpenAIProvider


def get_provider() -> LLMProvider:
    """Reads LLM_PROVIDER from settings and returns the matching implementation.
    This is the single switch point — swapping vendors never touches the RAG pipeline."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIProvider(settings.openai_api_key)

    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return GeminiProvider(settings.gemini_api_key)
