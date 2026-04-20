import pytest
from app.chunking import chunk_text_words

def test_chunking_basic_overlap():
    text = " ".join([f"w{i}" for i in range(0, 50)])  # 50 words
    chunks = chunk_text_words(text, chunk_size=20, overlap=5)

    # Starts at 0, 15, 30. The 30..50 chunk reaches the end, so we stop.
    assert len(chunks) == 3

    c0 = chunks[0].split()
    c1 = chunks[1].split()
    c2 = chunks[2].split()

    assert c0[-5:] == c1[:5]
    assert c1[-5:] == c2[:5]

def test_chunking_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text_words("a b c", chunk_size=10, overlap=10)