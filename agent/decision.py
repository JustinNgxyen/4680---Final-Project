def decide_action(query: str):
    query_lower = query.lower()

    if any(op in query for op in ["*", "+", "-", "/"]):
        return "calculator"

    if any(word in query_lower for word in ["what", "explain", "define"]):
        return "knowledge_base"

    return "final"