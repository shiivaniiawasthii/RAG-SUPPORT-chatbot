def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into overlapping word windows so an answer that spans
    a chunk boundary still has its full context in at least one chunk."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks
