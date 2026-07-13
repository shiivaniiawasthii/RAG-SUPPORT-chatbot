from openai import OpenAI

from app.llm.base import LLMProvider

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a support assistant. Answer the question using only the "
    "provided context. If the context doesn't contain the answer, say "
    "you don't have enough information — do not make things up."
)


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def generate(self, question: str, context: str) -> str:
        response = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content
