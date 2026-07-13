from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Every provider (Gemini, OpenAI, ...) implements this same shape.
    The rest of the app never imports a specific vendor SDK directly."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        ...
