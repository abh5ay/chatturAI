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
- 🎓 **Document-Grounded AI Tutor**: Ask complex academic questions; answers are grounded strictly in retrieved passages with exact 1-based page citations and an anti-hallucination guard.
- 📝 **Smart Notes Generator**: Generate structured study outlines (Short or Detailed) with concepts, key definitions, and revision takeaways.
- 💡 **Explain Simply**: Demystify difficult technical terms and formulas with everyday analogies and intuitive examples.
- 🎯 **Conceptual MCQ Quizzes**: Generate 5 challenging multiple-choice questions with 4 options each, validated by strict schema checking and scored locally in Python (answers hidden until submission).
- 🎴 **Active Recall Flashcards**: Generate up to 10 interactive revision cards with sequential navigation and answer-reveal toggles.
- 🔒 **Session-Isolated Architecture**: All vector embeddings and extracted materials reside in-memory per session; clearing documents thoroughly purges all dependent state.

---

## 🏗️ Architecture

```
PDF Upload
   ↓
Text Extraction (PyMuPDF)
   ↓
Chunking (RecursiveCharacterTextSplitter: 800 chars, 150 overlap)
   ↓
Sentence Embeddings (sentence-transformers/all-mpnet-base-v2)
   ↓
FAISS Vector Store (Session-Isolated In-Memory)
   ↓
Retrieval (Distance Filter: threshold 1.05, Top-K = 4)
   ↓
Groq LLM (qwen/qwen3.8-27b @ temp 0.0)
   ↓
Cited Answers / Study Tools
```

---

## 🛠️ Technology Stack

| Layer | Component | Description |
|:---|:---|:---|
| **Frontend** | Streamlit | Custom dark theme (`#0B0F19`), glassmorphic cards, responsive sidebar |
| **Parsing** | PyMuPDF (fitz) | Fast, reliable multi-page text extraction and metadata tracking |
| **Embeddings** | Sentence Transformers | `all-mpnet-base-v2` (768-dim, cached globally) |
| **Vector Store** | FAISS (CPU) | High-performance in-memory index isolated per user session |
| **Orchestration**| LangChain 0.2 | Document chains, prompt templates, and output validation |
| **LLM Inference**| Groq Cloud API | High-throughput `qwen/qwen3.8-27b` with tier-safe token limits |

---

## 🚀 Local Setup Instructions

### Prerequisites
- Python 3.12+
- Git
- Free [Groq API Key](https://console.groq.com/)

### 1. Clone & Navigate
```bash
git clone https://github.com/Ratnesh-181998/Universal-PDF-RAG-Chatbot.git chattur-ai
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

1. **Upload Notes**: Navigate to **📁 My Documents** and upload text-based PDF files. The system indexes passages and reports status.
2. **Consult AI Tutor**: Go to **🎓 AI Tutor** and ask natural language questions. View exact source references by expanding the **📚 Sources** panel.
3. **Generate Notes & Explanations**: Open **📝 Smart Notes** to synthesize comprehensive outlines or use **Explain Simply** for beginner-friendly analogies.
4. **Practice & Test**: Open **⚡ Practice** to take a 5-question multiple-choice quiz with immediate local grading, or drill concepts using active recall flashcards.
5. **Clear / Reset**: Use **🗑️ Clear All Documents** at any time to invalidate the session vector store and remove all dependent study artifacts.

---

## 🧪 Testing Instructions

Run the complete automated verification test suite:

```bash
# Run RAG & Retrieval Engine Tests
python -m unittest scratch/test_stage2_rag.py

# Run Streamlit UI & State Tests
python scratch/test_stage3_ui.py

# Run Study Tools (Notes, Explain Simply, Quiz, Flashcards) Tests
python -m unittest scratch/test_stage4_study_tools.py

# Run Security, Hardening & Upload Validation Tests
python -m unittest scratch/test_stage5_hardening.py
```

---

## ☁️ Deployment Instructions

### Option A: Streamlit Community Cloud (Recommended for Demo)
1. Fork or push the repository to GitHub.
2. Link your repository in [share.streamlit.io](https://share.streamlit.io).
3. Set main file path to `app.py`.
4. In **App settings → Secrets**, provide your API key:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Deploy.

> **Resource Note**: Streamlit Community Cloud has a 1GB RAM container limit. The `all-mpnet-base-v2` PyTorch model requires ~450MB RAM. For multi-tenant concurrency, container deployment (Option B) is recommended.

### Option B: Docker Container Deployment
Build and run the isolated container:
```bash
docker build -t chatturai:latest .
docker run -p 8501:8501 --env-file .env chatturai:latest
```

---

## ⚠️ Known Limitations

- **Scanned / Image-Only PDFs**: Text extraction relies on PyMuPDF text layers. Scanned PDFs without OCR emit a warning; OCR is out of scope for the current MVP.
- **Rate Limits**: Groq on-demand tier enforces Output Tokens Per Minute (OTPM) constraints. Token budgets are configured conservatively (`max_tokens=1000`).
- **Single-Node In-Memory Storage**: Vectors reside in Streamlit session memory. Refreshing the browser tab or closing the session destroys the ephemeral index.

---

## 📄 License & Attribution

- **License**: MIT License
- **Upstream Project**: Adapted from *Universal PDF RAG Chatbot* by Ratnesh Singh.
