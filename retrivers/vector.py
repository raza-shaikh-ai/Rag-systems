from langchain_pinecone import PineconeVectorStore
from embedding.embed import get_embeddings
import os

def get_vector_retriever(scope_id: str):
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=get_embeddings(),
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "filter": {"scope_id": scope_id}
        }
    )