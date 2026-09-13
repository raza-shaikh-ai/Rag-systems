from langchain_aws import BedrockEmbeddings
import os

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
    return _embeddings