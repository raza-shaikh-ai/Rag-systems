from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(docs):

    text_length = sum(len(doc.page_content) for doc in docs)
    if text_length < 2000:
        chunk_size = 300
        chunk_overlap = 50
    elif text_length < 10000:
        chunk_size = 600
        chunk_overlap = 100
    else:
        chunk_size = 900
        chunk_overlap = 150


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)