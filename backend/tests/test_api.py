import io

from fastapi.testclient import TestClient

import app.rag.pipeline as pipeline_module
from app.main import app

client = TestClient(app)


class FakeProvider:
    """Stands in for Gemini/OpenAI so tests run with no API key and no network."""

    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2]

    def generate(self, question, context):
        return f"answer to: {question}"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_chat(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_provider", lambda: FakeProvider())
    monkeypatch.setattr(pipeline_module.settings, "vector_store_dir", str(tmp_path))

    file_content = b"This is a support document about password resets."
    response = client.post(
        "/upload",
        files={"file": ("doc.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] == 1
    document_id = body["document_id"]

    chat_response = client.post(
        "/chat",
        json={"document_id": document_id, "question": "How do I reset my password?"},
    )
    assert chat_response.status_code == 200
    data = chat_response.json()
    assert "answer" in data
    assert len(data["sources"]) == 1


def test_chat_unknown_document_returns_404(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.get_provider", lambda: FakeProvider())
    monkeypatch.setattr(pipeline_module.settings, "vector_store_dir", str(tmp_path))

    response = client.post(
        "/chat", json={"document_id": "does-not-exist", "question": "anything"}
    )
    assert response.status_code == 404
