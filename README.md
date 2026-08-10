# 🧠 DocuMind — RAG-Powered Document Intelligence System
url : https://nangliaharsh-documind-frontendapp-3gg4d8.streamlit.app/

Production-grade multi-document RAG pipeline using LangChain, LLaMA 3 (Groq), ChromaDB and FastAPI enabling natural language Q&A across multiple documents with source-cited responses and conversational memory. Implemented hybrid retrieval combining BM25 sparse search with dense vector embeddings, followed by cross-encoder re-ranking and citation verification. Evaluated using RAGAS framework.

---

## ✨ Features

- 📄 **Multi-format ingestion** — PDF, DOCX, and TXT support
- 🔍 **Hybrid retrieval** — BM25 sparse search + dense vector embeddings
- 🎯 **Cross-encoder re-ranking** — precision-optimized context selection
- ✅ **Citation verification** — validates every source chunk against generated claims
- 🔄 **Query reformulation** — rewrites vague queries before retrieval
- 🧠 **Conversational memory** — context-aware multi-turn Q&A
- 📋 **Auto-summarization** — instant document summary on upload
- 📊 **Explainability panel** — chunk-level relevance scores per answer
- 📈 **RAGAS evaluation** — faithfulness and answer relevancy metrics

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3 via Groq API |
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace sentence-transformers |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluation | RAGAS |

---

## 🏗️ Architecture
User Query
↓
Query Reformulation (LLaMA 3.1 8b)
↓
Hybrid Search (BM25 + Semantic)
↓
Cross-Encoder Re-ranking
↓
Answer Generation (LLaMA 3.3 70b)
↓
Citation Verification Pass
↓
Answer + Sources + Explainability Panel

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/nangliaharsh/documind.git
cd documind
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:

### 5. Run the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Run the frontend
```bash
cd frontend
streamlit run app.py
```

### 7. Open in browser

---

## 📊 Evaluation

Run RAGAS evaluation on your documents:
```bash
cd backend
python run_eval.py
```

---

## 🔑 Get API Keys

- **Groq API** (free) — [console.groq.com](https://console.groq.com)

---

## 👤 Author

**Harsh Nanglia**
- GitHub: [@nangliaharsh](https://github.com/nangliaharsh)
- Email: Nangliaharsh@gmail.com
