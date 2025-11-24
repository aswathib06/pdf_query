def split_into_chunks(text, size=800, overlap=200):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks
