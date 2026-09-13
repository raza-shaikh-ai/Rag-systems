from langchain_pinecone import PineconeVectorStore
from embedding.embed import get_embeddings
from pinecone import Pinecone, ServerlessSpec
import os

def get_pinecone_index():
    """Ensure the Pinecone index exists, create if not."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=3072,     
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)


def store_chunks(chunks):
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
    get_pinecone_index()  # ensure index exists
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        index_name=index_name,
    )
    return vectorstore