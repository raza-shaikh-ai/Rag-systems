def rerank(query, docs):
    q_words = set(query.lower().split())
    scored = []

    for doc in docs:
        text = doc.page_content.lower()
        overlap = len(q_words & set(text.split()))
        scored.append((overlap, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]

def hybrid_retrieve_top1(query, bm25_retriever, vector_retriever, scope_id=None, k_each=5):
    #retrieve
    bm25_docs = bm25_retriever.invoke(query)[:k_each]
    vector_docs = vector_retriever.invoke(query)[:k_each]

    # scope filter
    if scope_id:
        bm25_docs = [d for d in bm25_docs if d.metadata.get("scope_id") == scope_id]
        vector_docs = [d for d in vector_docs if d.metadata.get("scope_id") == scope_id]

    #merge and deduplicate
    seen = set()
    merged = []
    for doc in bm25_docs + vector_docs:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    if not merged:
        return None
    # 4️⃣ re-rank
    ranked = rerank(query, merged)
    return ranked[0]

