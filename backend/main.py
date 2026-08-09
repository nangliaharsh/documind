import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
from utils import parse_file
from vector_store import store_chunks, get_collection
from rag_pipeline import query_documents
from rag_pipeline import query_documents, clear_memory, summarize_document

app = FastAPI(title="DocuMind API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

chat_history = []
uploaded_docs = []  # Track uploaded doc names

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    doc_ids: Optional[List[str]] = None  # None = search all docs

# Store summaries in memory
doc_summaries = {}

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Check file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type: {ext}. Allowed: PDF, DOCX, TXT"}

    upload_path = os.getenv("UPLOAD_PATH", "./data/uploads")
    os.makedirs(upload_path, exist_ok=True)
    file_path = f"{upload_path}/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = parse_file(file_path)
    doc_id = file.filename.replace(".pdf", "").replace(".docx", "").replace(".txt", "").replace(" ", "_")
    store_chunks(doc_id, chunks)

    if doc_id not in uploaded_docs:
        uploaded_docs.append(doc_id)

    # Auto generate summary
    summary = summarize_document(doc_id)
    doc_summaries[doc_id] = summary

    return {
        "doc_id": doc_id,
        "chunks_stored": len(chunks),
        "total_docs": len(uploaded_docs),
        "summary": summary
    }

@app.get("/summary/{doc_id}")
def get_summary(doc_id: str):
    summary = doc_summaries.get(doc_id, "Summary not available.")
    return {"doc_id": doc_id, "summary": summary}

@app.post("/query")
async def query(req: QueryRequest):
    try:
        answer, sources, explainability, citation_results, reformulation = query_documents(
            req.question, req.session_id, req.doc_ids
        )
        return {
            "answer": answer,
            "sources": sources,
            "explainability": explainability,
            "citation_results": citation_results,
            "reformulation": reformulation
        }
    except Exception as e:
        print(f"QUERY ERROR: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "explainability": [],
            "citation_results": [],
            "reformulation": {}
        }

@app.get("/documents")
def list_documents():
    return {"documents": uploaded_docs}

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id in uploaded_docs:
        uploaded_docs.remove(doc_id)
    return {"deleted": doc_id, "remaining": uploaded_docs}

@app.get("/health")
def health():
    return {"status": "ok"}