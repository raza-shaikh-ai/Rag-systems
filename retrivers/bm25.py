from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

all_chunks = []
# Initialize with dummy document to avoid empty list error
dummy_doc = Document(page_content="dummy", metadata={"scope_id": "dummy"})
bm25_retriever = BM25Retriever.from_documents([dummy_doc])
bm25_retriever.k = 5

def reinitialize_bm25():
    global bm25_retriever
    if all_chunks:
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = 5

def bm25_search(query: str, scope_id: str):
    docs = bm25_retriever.invoke(query)  # Fixed method name
    return [
        d for d in docs
        if d.metadata.get("scope_id") == scope_id
    ]