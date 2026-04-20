from app.retrieval import retrieve_top_k

async def search_knowledge(store, query: str, k: int = 4):
    results = await retrieve_top_k(store, query, k)
    return results