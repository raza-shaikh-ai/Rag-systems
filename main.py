import os
from pathlib import Path
import hashlib
from fastapi import FastAPI,Depends,UploadFile, Form
from auth.dependencies import get_scope_id
from auth.jwtutils import create_scope_token
from rewrite.queryrewrite import rewrite
from simpleai import simple_chain
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"

os.environ.setdefault("HF_HOME", str(MODEL_DIR))
TEMP_DIR.mkdir(exist_ok=True)
from inject import inject

app = FastAPI()

@app.post("/login")
def login(username: str):
    # deterministic scope
    scope_id = hashlib.sha256(username.lower().encode()).hexdigest()

    token = create_scope_token(scope_id)

    return {"token": token}

@app.post("/inject/file")
async def inject_file(
    file: UploadFile,
    scope_id: str = Depends(get_scope_id)
):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        source_type = 'pdf'
    elif filename.endswith('.csv'):
        source_type = 'csv'
    elif filename.endswith('.docx'):
        source_type = 'docx'
    elif filename.endswith('.txt'):
        source_type = 'txt'
    else:
        return {"error": f"Unsupported file type. Supported: .pdf, .csv, .docx, .txt"}

    # Use absolute path
    path = TEMP_DIR / file.filename
    try:
        with open(path, "wb") as f:
            f.write(await file.read())

        inject(source_type, str(path), scope_id)
        return {"status": "Done loading"}
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}
    finally:
        if path.exists():
            path.unlink()

from rag import rag_chain
from retrivers.hybrid import hybrid_retrieve_top1
from retrivers.bm25 import bm25_retriever
from retrivers.vector import get_vector_retriever

@app.post("/ask")
async def ask(
    question: str,
    scope_id: str = Depends(get_scope_id)
):
    query_rewriting = question #rewrite require higher ""rewrite(question)"" compuation
    vector_retriever = get_vector_retriever(scope_id)
    top_doc = hybrid_retrieve_top1(query_rewriting, bm25_retriever, vector_retriever, scope_id)

    if not top_doc or len(top_doc.page_content.strip()) < 10:
        return {"answer": simple_chain.invoke({"question": query_rewriting})}

    answer = rag_chain.invoke({
        "context": top_doc.page_content,
        "question": query_rewriting
    })

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)