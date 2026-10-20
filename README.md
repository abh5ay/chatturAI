# 🎓 chatturAI — Intelligent Study Assistant

> **"Ratta Nahi, Samajh Ke Padho!"**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green.svg)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**chatturAI** is a privacy-first, retrieval-augmented study assistant designed to help students master academic concepts rather than memorizing blindly. Upload course syllabi, textbooks, or lecture slides (PDF) and instantly unlock interactive AI tutoring, structured notes, analogies, quizzes, and revision cards—all strictly grounded in your materials with verifiable page citations.

---

## 🌟 Features

- 📁 **Multi-PDF Document Hub**: Upload up to 10 course PDFs (up to 50MB each) with instant SHA-256 deduplication and page-level metadata indexing.
- 👁️ **Smart OCR for Scanned Documents**: Automatic text vs. scan detection via PyMuPDF and local Tesseract OCR. Handles searchable, scanned, and mixed pages without text duplication while preserving exact page citations.
- 🗂️ **Persistent Study Workspaces**: Create, switch, rename, and delete isolated workspaces. Workspaces persist documents, FAISS vector indices, chat history, notes, quizzes, and flashcards across restarts with instant (<2ms) reloading and zero re-embedding.
- 🎓 **Document-Grounded AI Tutor**: Ask complex academic questions; answers are grounded strictly in retrieved passages with exact 1-based page citations and an anti-hallucination guard.
- ⚡ **Intelligent Groq Optimization**: Centralized resilient execution with 429 `Retry-After` parsing, exponential backoff, feature-specific token budgets, and session cache reuse to eliminate redundant API calls.
- 📝 **Smart Notes Generator**: Synthesize structured study outlines (Short or Detailed) with concepts, key definitions, and revision takeaways.
- 💡 **Explain Simply**: Demystify difficult technical terms and formulas with everyday analogies and intuitive examples.
- 🎯 **Conceptual MCQ Quizzes**: Generate 5 challenging multiple-choice questions with 4 options each, validated by strict schema checking and scored locally in Python (answers hidden until submission).
- 🎴 **Active Recall Flashcards**: Interactive revision cards with sequential navigation and answer-reveal toggles.
- 🚀 **Fast Embeddings**: Pure PyTorch + SafeTensors loader for `all-mpnet-base-v2`, reducing app startup times by over 90% without heavy transformer library overhead.

---

## 🏗️ Architecture

```
[ User PDF Upload ] ──► [ Page Classifier: Searchable vs Image vs Mixed ]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [ PyMuPDF Text Layer ]             [ Tesseract OCR Engine ]
                  └─────────────────┬─────────────────┘
                                    ▼
                    [ Recursive Character Splitter ]
                     (Chunk: 800, Overlap: 150)
                                    ▼
                 [ Fast MPNet SafeTensors Embeddings ]
                                    ▼
                   [ Isolated FAISS Vector Index ]
                                    ▼
            [ SQLite WAL Metadata & Workspace Persistence ]
                                    ▼
         [ Hybrid Retrieval (Keyword Match + Vector Cosine) ]
                                    ▼
       [ Resilient Groq API Runner (Adaptive Backoff & Cache) ]
                                    ▼
          ┌─────────────────────────┴─────────────────────────┐
          ▼                         ▼                         ▼
   [ AI Tutor Chat ]        [ Smart Study Notes ]      [ Quizzes & Cards ]
```

---

## 🛠️ Technology Stack

| Layer | Component | Description |
|:---|:---|:---|
| **Frontend** | Streamlit | Custom dark theme (`#0B0F19`), glassmorphic cards, responsive sidebar |
| **Parsing & OCR** | PyMuPDF + Tesseract | Hybrid searchable text extraction and local optical character recognition |
| **Embeddings** | Fast MPNet SafeTensors | `all-mpnet-base-v2` (768-dim, sub-second initialization) |
| **Vector Store** | FAISS (CPU) | High-performance vector index persisted atomically per workspace |
| **Persistence** | SQLite (WAL mode) | Metadata, chunk tracking, conversation history, and study feature states |
| **Orchestration**| LangChain 0.2 | Document chains, prompt templates, and output validation |
| **LLM Inference**| Groq Cloud API | High-throughput `qwen/qwen3.8-27b` with resilient rate-limit handling |

---

## 🚀 Local Setup Instructions

### Prerequisites
- Python 3.12+
- Git
- Free [Groq API Key](https://console.groq.com/)
- *(Optional, for OCR on scanned PDFs)*: Tesseract OCR engine

#### Installing Tesseract OCR (Optional for scanned PDFs)
- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
- **Windows**: Download and install from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki), then add to PATH.

*(Note: chatturAI automatically checks for Tesseract. If absent, regular text PDFs will continue to process with zero issues.)*

### 1. Clone & Navigate
```bash
git clone https://github.com/abh5ay/chatturAI.git chattur-ai
cd chattur-ai
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Edit `.env` and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(The `.env` file is gitignored to protect your credentials.)*

### 5. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📖 Usage Guide

1. **Manage Workspaces**: Use the sidebar to create dedicated workspaces (e.g., "Operating Systems", "Linear Algebra"). Workspaces save your uploaded PDFs, index, chat history, notes, and quiz state.
2. **Upload Notes**: Navigate to **📁 My Documents** and upload your PDFs. If a page has scanned or handwritten images, OCR will automatically extract the text.
3. **Consult AI Tutor**: Go to **🎓 AI Tutor** and ask natural language questions. View exact source references by expanding the **📚 Sources** panel.
4. **Generate Notes & Explanations**: Open **📝 Smart Notes** to synthesize comprehensive outlines or use **Explain Simply** for beginner-friendly analogies.
5. **Practice & Test**: Open **⚡ Practice** to take a 5-question multiple-choice quiz with immediate local grading, or drill concepts using active recall flashcards.
6. **Persistence**: Close your browser or restart the app anytime. Select your workspace to resume instantly with 0 re-indexing and 0 external API calls.

---

## 🧪 Testing Instructions

Run the automated verification test suites:

```bash
# Verify Workspace Persistence & Storage Isolation
python -c "from workspace_manager import WorkspaceManager; m = WorkspaceManager(); print(f'Workspaces: {len(m.list_workspaces())}')"

# Test FastMPNet Embeddings and Vector Search
python -c "from rag_engine import load_fast_embeddings; e = load_fast_embeddings(); print(f'Embedding dim: {len(e.embed_query(\"test\"))}')"
```

---

## 🔐 Security & Privacy

- ✅ API keys stored strictly in `.env` (gitignored).
- ✅ Private documents and vector stores saved locally in `workspaces_storage/` (gitignored).
- ✅ SQLite databases and journal files excluded from git.
- ✅ In-memory sanitization of file paths to prevent directory traversal.
- ✅ Zero external tracking or analytics.

---

## 📄 License & Attribution

- **License**: MIT License — see [LICENSE](LICENSE) for details.
- **Upstream Reference**: Originally adapted from *Universal PDF RAG Chatbot* by Ratnesh Singh.
