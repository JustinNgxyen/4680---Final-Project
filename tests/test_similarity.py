from app.similarity import cosine_similarity

def test_cosine_similarity_ranking():
    q = [1.0, 0.0]
    a = [1.0, 0.0]   # perfect match
    b = [0.5, 0.5]   # partial match
    c = [0.0, 1.0]   # orthogonal

    sa = cosine_similarity(q, a)
    sb = cosine_similarity(q, b)
    sc = cosine_similarity(q, c)

    assert sa > sb > sc
    assert sa == 1.0
    assert sc == 0.0