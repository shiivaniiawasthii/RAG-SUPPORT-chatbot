from google import genai

from app.llm.base import LLMProvider

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = (
    "You are a support assistant. Answer the question using only the "
    "provided context. If the context doesn't contain the answer, say "
    "you don't have enough information — do not make things up."
)


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        result = self._client.models.embed_content(model=EMBED_MODEL, contents=text)
        return result.embeddings[0].values

    def generate(self, question: str, context: str) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        response = self._client.models.generate_content(model=CHAT_MODEL, contents=prompt)
        return response.text
