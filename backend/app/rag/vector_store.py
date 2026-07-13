import json
import os

import faiss
import numpy as np


class VectorStore:
    """One FAISS index per uploaded document, persisted to disk under
    <base_dir>/<document_id>/. Keeps the demo stateless between requests
    without needing an external vector database."""

    def __init__(self, base_dir: str, document_id: str):
        self._dir = os.path.join(base_dir, document_id)
        self._index_path = os.path.join(self._dir, "index.faiss")
        self._meta_path = os.path.join(self._dir, "meta.json")

    def save(self, embeddings: list[list[float]], chunks: list[str], filename: str) -> None:
        os.makedirs(self._dir, exist_ok=True)
        matrix = np.array(embeddings, dtype="float32")
        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, self._index_path)

        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({"filename": filename, "chunks": chunks}, f)

    def exists(self) -> bool:
        return os.path.exists(self._index_path) and os.path.exists(self._meta_path)

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        if not self.exists():
            raise FileNotFoundError("Document not found")

        index = faiss.read_index(self._index_path)
        with open(self._meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        query = np.array([query_embedding], dtype="float32")
        distances, indices = index.search(query, min(top_k, index.ntotal))

        chunks = meta["chunks"]
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((chunks[idx], float(dist)))
        return results
