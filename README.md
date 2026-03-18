# 🧠 Local Enterprise RAG System

A **100% locally hosted** Retrieval-Augmented Generation (RAG) platform. This system allows you to securely upload multiple document types, perform intelligent semantic analysis, and interact with the content via a conversational AI interface. **No data leaves your local machine, ensuring total privacy.**

## 🚀 Key Features

- **Multi-Format Support:** Ingest `PDF`, `DOCX`, `TXT`, `MD`, `CSV`, `XLSX`, `PPTX`, and `HTML`.
- **Intelligent Chunking:** Semantically splits documents while preserving page structure, headings, and tables.
- **Hybrid Search Engine:** Combines **FAISS** (dense semantic vectors via `BAAI/bge-m3`) with **BM25** (sparse keyword index) for maximum retrieval accuracy.
- **Cross-Encoder Reranking:** Uses `ms-marco-MiniLM-L-6-v2` to rerank the top 20 results down to the absolute best 3-5 chunks.
- **Visual Question Answering (VQA):** Integrates **LLaVA 7b** to analyze and describe charts, graphs, and image-heavy pages inside PDFs.
- **Context-Aware Chat:** Powered locally by **Llama 3.2**, answering complex queries with precise inline citations (e.g., `[SOURCE: Report.pdf | Page 4]`).
- **Short-Term Memory (STM):** Efficiently manages conversation history to prevent "context exhaustion" and keep token costs low.

## 🛠️ Technology Stack

- **Backend:** Python, FastAPI, FAISS, BM25, SQLite
- **Frontend:** Streamlit 
- **AI Engine:** Ollama
- **Models Used:** 
  - `llama3.2` (Text generation & reasoning)
  - `llava:7b` (Vision & chart analysis)
  - `BAAI/bge-m3` (Embeddings)
  - `ms-marco-MiniLM-L-6-v2` (Reranker)

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed on your machine.

### 2. Install AI Models
Run the following commands in your terminal to pull the required local models:
```bash
ollama pull llama3.2
ollama pull llava:7b
```

### 3. Setup the Environment
Clone the repository and install the Python dependencies:
```bash
git clone https://github.com/your-username/Local-Enterprise-RAG.git
cd Local-Enterprise-RAG

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Running the Application
The system requires both the backend API and frontend UI to be running simultaneously.

**Terminal 1 — Backend:**
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
python -m streamlit run frontend/app.py --server.port 8501
```

Open your browser to `http://localhost:8501`.

## 🔒 Privacy & Security

This architecture is completely air-gapped from cloud APIs. The entire vector database (FAISS/BM25) and all LLM inferences occur locally on your machine's CPU/GPU, eliminating cloud compute costs and preventing sensitive enterprise data leakage.
