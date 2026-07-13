from app.rag.chunking import chunk_text


def test_chunk_text_basic():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    assert chunks[0] == "word0 word1 word2 word3"
    assert chunks[1].startswith("word3")


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=10, overlap=2) == []


def test_chunk_text_shorter_than_chunk_size():
    text = "just five short words"
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks == ["just five short words"]
