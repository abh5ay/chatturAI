import datetime
import html
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Prevent HuggingFace network retries and OpenMP conflicts on macOS
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env", override=True)

# Core RAG engine components
from rag_engine import (
    process_pdf_files,
    build_isolated_vectorstore,
    retrieve_chunks,
    format_citations_metadata,
    get_rag_prompt_template,
    load_fast_embeddings,
    INSUFFICIENT_EVIDENCE_MSG,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)
from study_tools import (
    generate_smart_notes,
    explain_simply,
    generate_mcq_quiz,
    generate_flashcards,
    invoke_llm_resilient,
    TOKEN_BUDGETS,
    compute_study_cache_key,
)
from workspace_manager import WorkspaceManager

# LangChain & Vectorstore components
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    from langchain_community.llms import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Application Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = DEFAULT_CHUNK_SIZE
CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP
TOP_K = DEFAULT_TOP_K

GROQ_MODELS = [
    "qwen/qwen3.8-27b",            # Primary high-performance model
    "openai/gpt-oss-120b",         # Active 120B model
    "openai/gpt-oss-20b",          # Active fast model
    "llama-3.3-70b-versatile",     # Llama 3.3 70B
    "llama-3.1-8b-instant",        # Fast & efficient
]

# Logging Setup
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(message: str, level: str = "INFO"):
    """Log an event to file and session state without printing secrets."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
        
    if "app_logs" not in st.session_state:
        st.session_state["app_logs"] = []
    
    emoji = '🔴' if level == 'ERROR' else ('⚠️' if level == 'WARNING' else '🔹')
    st.session_state["app_logs"].insert(0, f"{emoji} {log_entry}")
    if len(st.session_state["app_logs"]) > 100:
        st.session_state["app_logs"] = st.session_state["app_logs"][:100]

# Page Configuration
st.set_page_config(
    page_title="chatturAI — Intelligent Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System (Modern Educational SaaS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Global App Background */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #13112c 0%, #0b0f19 50%, #07090e 100%);
        color: #f3f4f6;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0d111d;
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    
    /* Header & Tagline */
    .brand-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .brand-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #c084fc 0%, #818cf8 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .brand-tagline {
        font-size: 1.15rem;
        font-weight: 700;
        color: #a78bfa;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .brand-subheading {
        color: #94a3b8;
        font-size: 0.95rem;
    }
    
    /* Card Component */
    .chattur-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    /* Metric Badge */
    .stat-badge {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #e2e8f0;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    
    /* Primary Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%);
        color: #ffffff !important;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        border: 1px solid rgba(167, 139, 250, 0.3);
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
        border-color: rgba(192, 132, 252, 0.6);
    }
    
    /* Secondary/Danger Buttons */
    button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
    }
    
    /* Citation Expander */
    .stExpander {
        background: rgba(17, 24, 39, 0.6) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 12px !important;
        margin-top: 0.8rem;
    }
    
    /* Coming Soon Badge */
    .coming-soon-badge {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Cached Embedding Initializer (Safe global cache)
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load and cache fast MPNet embedding model."""
    return load_fast_embeddings(EMBEDDING_MODEL_NAME)

def get_groq_llm_instance(api_key: str, model_name: str, max_tokens: int = 1000):
    """Instantiate a ChatGroq LLM with deterministic low-temperature and tier-safe token budget."""
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.0,
        max_tokens=max_tokens
    )

def get_default_study_llm(max_tokens: int = 1000):
    """Retrieve configured ChatGroq instance for study features with token budget."""
    groq_key = os.environ.get("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else "")
    if GROQ_AVAILABLE and groq_key:
        return get_groq_llm_instance(groq_key, "qwen/qwen3.8-27b", max_tokens=max_tokens)
    openai_key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else "")
    if OPENAI_AVAILABLE and openai_key:
        return OpenAI(temperature=0.0, openai_api_key=openai_key)
    return None

# Session State Initialization
if "feature_cache" not in st.session_state:
    st.session_state["feature_cache"] = {}

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None
if "uploaded_file_info" not in st.session_state:
    st.session_state["uploaded_file_info"] = []
if "processed_file_keys" not in st.session_state:
    st.session_state["processed_file_keys"] = []
if "total_chunks" not in st.session_state:
    st.session_state["total_chunks"] = 0
if "document_chunks" not in st.session_state:
    st.session_state["document_chunks"] = []
if "nav_section" not in st.session_state:
    st.session_state["nav_section"] = "🏠 Home"

# Stage 4: Study Features Session State
if "smart_notes" not in st.session_state:
    st.session_state["smart_notes"] = None
if "notes_doc_key" not in st.session_state:
    st.session_state["notes_doc_key"] = None
if "explain_result" not in st.session_state:
    st.session_state["explain_result"] = None
if "current_quiz" not in st.session_state:
    st.session_state["current_quiz"] = None
if "quiz_doc_key" not in st.session_state:
    st.session_state["quiz_doc_key"] = None
if "quiz_answers" not in st.session_state:
    st.session_state["quiz_answers"] = {}
if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False
if "quiz_score" not in st.session_state:
    st.session_state["quiz_score"] = None
if "flashcards" not in st.session_state:
    st.session_state["flashcards"] = None
if "flashcards_doc_key" not in st.session_state:
    st.session_state["flashcards_doc_key"] = None
if "flashcard_idx" not in st.session_state:
    st.session_state["flashcard_idx"] = 0
if "flashcard_show_back" not in st.session_state:
    st.session_state["flashcard_show_back"] = False

# ------------------------------------------------------------------------------
# PHASE 3: PERSISTENT STUDY WORKSPACE INITIALIZATION
# ------------------------------------------------------------------------------
ws_manager = WorkspaceManager()
all_workspaces = ws_manager.list_workspaces()
if not all_workspaces:
    default_ws = ws_manager.create_workspace("Default Workspace")
    all_workspaces = [default_ws]

if "active_workspace_id" not in st.session_state:
    st.session_state["active_workspace_id"] = all_workspaces[0]["id"]
    loaded_init = ws_manager.load_workspace_state(st.session_state["active_workspace_id"], get_embeddings())
    if loaded_init:
        for k, v in loaded_init.items():
            if k not in ["id", "name"]:
                st.session_state[k] = v


def sync_current_workspace(files_data=None):
    ws_id = st.session_state.get("active_workspace_id")
    if ws_id:
        try:
            ws_manager.save_workspace_state(
                workspace_id=ws_id,
                uploaded_file_info=st.session_state.get("uploaded_file_info", []),
                document_chunks=st.session_state.get("document_chunks", []),
                vector_db=st.session_state.get("vector_db"),
                messages=st.session_state.get("messages", []),
                smart_notes=st.session_state.get("smart_notes"),
                explain_result=st.session_state.get("explain_result"),
                current_quiz=st.session_state.get("current_quiz"),
                quiz_state={
                    "quiz_answers": st.session_state.get("quiz_answers", {}),
                    "quiz_submitted": st.session_state.get("quiz_submitted", False),
                    "quiz_score": st.session_state.get("quiz_score"),
                },
                flashcards=st.session_state.get("flashcards"),
                flashcard_idx=st.session_state.get("flashcard_idx", 0),
                feature_cache=st.session_state.get("feature_cache", {}),
                files_data=files_data,
            )
        except Exception as e:
            log_event(f"Workspace sync error: {e}", "WARNING")


# Sidebar Navigation & System Status
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size: 2.5rem; margin-bottom: 0.2rem;'>🎓</div>
        <h2 style='margin: 0; font-size: 1.4rem; color: #f8fafc;'>chatturAI</h2>
        <p style='margin: 0; font-size: 0.8rem; color: #a78bfa; font-weight: 600;'>Ratta Nahi, Samajh Ke Padho!</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Workspace Selection & Management (Phase 3)
    st.markdown("#### 🗂️ Study Workspace")
    ws_id_to_name = {w["id"]: w["name"] for w in all_workspaces}
    ws_ids = [w["id"] for w in all_workspaces]
    current_ws_id = st.session_state.get("active_workspace_id", ws_ids[0])
    current_ws_idx = ws_ids.index(current_ws_id) if current_ws_id in ws_ids else 0

    selected_ws_id = st.selectbox(
        "Active Workspace",
        options=ws_ids,
        format_func=lambda wid: f"{ws_id_to_name.get(wid, wid)}",
        index=current_ws_idx,
        key="active_workspace_select",
        label_visibility="collapsed"
    )

    if selected_ws_id != st.session_state.get("active_workspace_id"):
        sync_current_workspace()
        st.session_state["active_workspace_id"] = selected_ws_id
        st.session_state["vector_db"] = None
        st.session_state["uploaded_file_info"] = []
        st.session_state["processed_file_keys"] = []
        st.session_state["total_chunks"] = 0
        st.session_state["document_chunks"] = []
        st.session_state["feature_cache"] = {}
        st.session_state["smart_notes"] = None
        st.session_state["explain_result"] = None
        st.session_state["current_quiz"] = None
        st.session_state["flashcards"] = None
        st.session_state["messages"] = []
        loaded = ws_manager.load_workspace_state(selected_ws_id, get_embeddings())
        if loaded:
            for k, v in loaded.items():
                if k not in ["id", "name"]:
                    st.session_state[k] = v
        log_event(f"Switched to workspace: {ws_id_to_name.get(selected_ws_id)}")
        st.rerun()

    with st.expander("⚙️ Manage Workspaces", expanded=False):
        new_ws_name = st.text_input("New workspace name:", placeholder="e.g. Operating Systems", key="new_ws_input")
        if st.button("➕ Create Workspace", use_container_width=True, disabled=not new_ws_name.strip()):
            created = ws_manager.create_workspace(new_ws_name.strip())
            st.session_state["active_workspace_id"] = created["id"]
            st.session_state["vector_db"] = None
            st.session_state["uploaded_file_info"] = []
            st.session_state["processed_file_keys"] = []
            st.session_state["total_chunks"] = 0
            st.session_state["document_chunks"] = []
            st.session_state["feature_cache"] = {}
            st.session_state["smart_notes"] = None
            st.session_state["explain_result"] = None
            st.session_state["current_quiz"] = None
            st.session_state["flashcards"] = None
            st.session_state["messages"] = []
            st.success(f"Created '{created['name']}'!")
            st.rerun()

        rename_name = st.text_input("Rename current workspace:", value=ws_id_to_name.get(selected_ws_id, ""), key="rename_ws_input")
        if st.button("✏️ Rename Workspace", use_container_width=True, disabled=not rename_name.strip()):
            if ws_manager.rename_workspace(selected_ws_id, rename_name.strip()):
                st.success("Workspace renamed!")
                st.rerun()

        if len(all_workspaces) > 1:
            confirm_del = st.checkbox("Confirm deletion of this workspace", key="confirm_del_ws")
            if st.button("🗑️ Delete Workspace", use_container_width=True, disabled=not confirm_del):
                ws_manager.delete_workspace(selected_ws_id)
                remaining = ws_manager.list_workspaces()
                st.session_state["active_workspace_id"] = remaining[0]["id"]
                loaded = ws_manager.load_workspace_state(remaining[0]["id"], get_embeddings())
                if loaded:
                    for k, v in loaded.items():
                        if k not in ["id", "name"]:
                            st.session_state[k] = v
                st.success("Workspace deleted.")
                st.rerun()

    st.markdown("---")
    
    # Navigation Radio
    nav_options = [
        "🏠 Home",
        "📁 My Documents",
        "🎓 AI Tutor",
        "📝 Smart Notes",
        "⚡ Practice",
    ]
    current_idx = nav_options.index(st.session_state["nav_section"]) if st.session_state["nav_section"] in nav_options else 0
    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=current_idx,
        key="sidebar_nav_selection"
    )
    st.session_state["nav_section"] = selected_nav
    
    st.markdown("---")
    
    # System & API Status
    groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    has_groq = bool(groq_key)
    doc_count = len(st.session_state.get("uploaded_file_info", []))
    chunk_count = st.session_state.get("total_chunks", 0)
    
    st.markdown("#### ⚡ System Status")
    if has_groq:
        st.markdown("🟢 **Groq LLM**: `Connected`")
        st.caption(f"Model: `{GROQ_MODELS[0]}`")
    else:
        st.markdown("🔴 **Groq LLM**: `Key Missing`")
        st.caption("Add `GROQ_API_KEY` to `.env` to enable AI answers.")
        
    st.markdown(f"📁 **Indexed Docs**: `{doc_count}`")
    st.markdown(f"🧩 **Active Chunks**: `{chunk_count}`")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 0.75rem; color: #64748b; padding-top: 1rem;'>
        chatturAI v2.0 • Study Assistant<br>
        Built with LangChain, FAISS & Groq
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION A: HOME DASHBOARD
# ==============================================================================
if st.session_state["nav_section"] == "🏠 Home":
    st.markdown("""
    <div class='brand-header'>
        <div class='brand-title'>Welcome to chatturAI 🎓</div>
        <div class='brand-tagline'>"Ratta Nahi, Samajh Ke Padho!"</div>
        <div class='brand-subheading'>Your notes. Your questions. Smarter learning.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time Metrics Dashboard
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(f"""
        <div class='stat-badge'>
            <div class='stat-number'>{len(st.session_state.get("uploaded_file_info", []))}</div>
            <div class='stat-label'>Uploaded Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class='stat-badge'>
            <div class='stat-number'>{st.session_state.get("total_chunks", 0)}</div>
            <div class='stat-label'>Indexed Passages</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        status_text = "Online ✅" if has_groq else "Setup Required ⚠️"
        st.markdown(f"""
        <div class='stat-badge'>
            <div class='stat-number' style='font-size: 1.4rem; padding-top: 0.4rem;'>{status_text}</div>
            <div class='stat-label'>AI Tutor Status</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Navigation Actions
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("📁 Upload & Manage Notes", use_container_width=True):
            st.session_state["nav_section"] = "📁 My Documents"
            st.rerun()
    with col_act2:
        if st.button("🎓 Launch AI Study Tutor", use_container_width=True):
            st.session_state["nav_section"] = "🎓 AI Tutor"
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Start Guide Card
    st.markdown("""
    <div class='chattur-card'>
        <h3 style='margin-top: 0; color: #c084fc;'>🚀 Quick-Start Guide</h3>
        <p style='margin-bottom: 0.8rem;'>Study effectively using your own course materials in three simple steps:</p>
        <ol style='padding-left: 1.2rem; line-height: 1.8;'>
            <li><strong>Upload Documents</strong>: Head to <em>My Documents</em> and drag-and-drop your textbooks, syllabi, or lecture slides (PDF).</li>
            <li><strong>Semantic Indexing</strong>: PyMuPDF and FAISS extract text and create isolated vector embeddings locally.</li>
            <li><strong>Ask Your AI Tutor</strong>: Ask questions in plain language and receive clear conceptual explanations grounded directly in your document with page citations.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards Grid
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("""
        <div class='chattur-card'>
            <h4 style='margin-top: 0; color: #818cf8;'>📁 My Documents</h4>
            <p style='font-size: 0.9rem;'>Upload, inspect, and organize textbooks and class notes. Includes instant deduplication and page-level metadata tracking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='chattur-card'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <h4 style='margin: 0; color: #a855f7;'>📝 Smart Notes</h4>
                <span class='coming-soon-badge' style='background: rgba(168, 85, 247, 0.15); color: #c084fc; border-color: rgba(168, 85, 247, 0.4);'>READY ✅</span>
            </div>
            <p style='font-size: 0.9rem; margin-top: 0.6rem;'>Automated concept summaries, simple explanations, and high-yield revision outlines generated from your materials.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_f2:
        st.markdown("""
        <div class='chattur-card'>
            <h4 style='margin-top: 0; color: #38bdf8;'>🎓 AI Tutor</h4>
            <p style='font-size: 0.9rem;'>Interactive study partner answering questions strictly from your texts. No made-up facts; transparent page number citations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='chattur-card'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <h4 style='margin: 0; color: #38bdf8;'>⚡ Practice & Recall</h4>
                <span class='coming-soon-badge' style='background: rgba(56, 189, 248, 0.15); color: #38bdf8; border-color: rgba(56, 189, 248, 0.4);'>READY ✅</span>
            </div>
            <p style='font-size: 0.9rem; margin-top: 0.6rem;'>Active recall flashcards and adaptive 5-question multiple-choice quizzes generated directly from your uploaded chapters.</p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION B: MY DOCUMENTS
# ==============================================================================
elif st.session_state["nav_section"] == "📁 My Documents":
    st.markdown("## 📁 My Documents")
    st.markdown("Upload textbooks, syllabi, and study notes. Supported: Text-based PDFs up to 200MB.")
    
    # PDF File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Study Materials",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to create your session knowledge base",
        key="doc_file_uploader"
    )
    
    col_proc1, col_proc2 = st.columns([3, 1])
    with col_proc1:
        if uploaded_files:
            current_file_keys = [f"{f.name}_{len(f.getvalue())}" for f in uploaded_files]
            
            # Reprocess only when files change or index not built
            if (st.session_state.get("processed_file_keys") != current_file_keys 
                or st.session_state.get("vector_db") is None):
                
                with st.spinner("📖 Ingesting documents with PyMuPDF & indexing passages..."):
                    file_tuples = [(f.name, f.getvalue()) for f in uploaded_files]
                    ingest_result = process_pdf_files(
                        file_tuples,
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                    
                    for err in ingest_result.errors:
                        st.error(f"❌ {err}")
                        log_event(f"Ingestion error: {err}", "ERROR")
                    for warn in ingest_result.warnings:
                        st.warning(f"⚠️ {warn}")
                        log_event(f"Ingestion warning: {warn}", "WARNING")
                    for dup in ingest_result.skipped_duplicates:
                        st.info(f"ℹ️ Skipped duplicate: {dup}")
                        
                    if ingest_result.documents:
                        embeddings = get_embeddings()
                        vector_db = build_isolated_vectorstore(ingest_result.documents, embeddings)
                        
                        # Store session state
                        st.session_state["vector_db"] = vector_db
                        st.session_state["processed_file_keys"] = current_file_keys
                        st.session_state["total_chunks"] = len(ingest_result.documents)
                        st.session_state["document_chunks"] = ingest_result.documents
                        
                        # Invalidate stale study state and response cache on new ingestion
                        st.session_state["feature_cache"] = {}
                        st.session_state["smart_notes"] = None
                        st.session_state["notes_doc_key"] = None
                        st.session_state["explain_result"] = None
                        st.session_state["current_quiz"] = None
                        st.session_state["quiz_doc_key"] = None
                        st.session_state["quiz_answers"] = {}
                        st.session_state["quiz_submitted"] = False
                        st.session_state["quiz_score"] = None
                        st.session_state["flashcards"] = None
                        st.session_state["flashcards_doc_key"] = None
                        st.session_state["flashcard_idx"] = 0
                        st.session_state["flashcard_show_back"] = False
                        
                        # Record document metadata
                        file_info_list = []
                        files_bytes_dict = {}
                        for f in uploaded_files:
                            b = f.getvalue()
                            files_bytes_dict[f.name] = b
                            file_info_list.append({
                                "name": f.name,
                                "size_kb": round(len(b) / 1024, 1),
                                "status": "Indexed & Ready ✅"
                            })
                        st.session_state["uploaded_file_info"] = file_info_list
                        sync_current_workspace(files_data=files_bytes_dict)
                        st.success(f"✅ Successfully indexed {len(ingest_result.documents)} passages from {len(ingest_result.processed_files)} document(s)!")
                        log_event(f"Indexed {len(ingest_result.documents)} chunks across {len(ingest_result.processed_files)} document(s)")
                    else:
                        st.error("❌ No readable text could be extracted from the uploaded files.")
    
    with col_proc2:
        if st.session_state.get("vector_db") is not None:
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                st.session_state["vector_db"] = None
                st.session_state["uploaded_file_info"] = []
                st.session_state["processed_file_keys"] = []
                st.session_state["total_chunks"] = 0
                st.session_state["document_chunks"] = []
                st.session_state["feature_cache"] = {}
                st.session_state["smart_notes"] = None
                st.session_state["notes_doc_key"] = None
                st.session_state["explain_result"] = None
                st.session_state["current_quiz"] = None
                st.session_state["quiz_doc_key"] = None
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_score"] = None
                st.session_state["flashcards"] = None
                st.session_state["flashcards_doc_key"] = None
                st.session_state["flashcard_idx"] = 0
                st.session_state["flashcard_show_back"] = False
                sync_current_workspace()
                st.success("All documents cleared and vector store invalidated.")
                log_event("Documents cleared by user.")
                st.rerun()
                
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display Active Documents Table
    active_docs = st.session_state.get("uploaded_file_info", [])
    if active_docs:
        st.markdown("### 📚 Active Study Materials")
        table_md = "| # | Document Name | File Size | Status |\n|:---:|:---|:---:|:---:|\n"
        for i, doc in enumerate(active_docs, 1):
            table_md += f"| **{i}** | 📄 `{doc['name']}` | {doc['size_kb']} KB | {doc['status']} |\n"
        st.markdown(table_md)
    else:
        st.markdown("""
        <div class='chattur-card' style='text-align: center; padding: 3rem;'>
            <div style='font-size: 3rem; margin-bottom: 0.8rem;'>📂</div>
            <h3 style='color: #a78bfa; margin-top: 0;'>No Study Materials Uploaded</h3>
            <p style='color: #94a3b8;'>Upload your course PDFs above to enable your personalized AI Tutor.</p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION C: AI TUTOR
# ==============================================================================
elif st.session_state["nav_section"] == "🎓 AI Tutor":
    st.markdown("## 🎓 AI Tutor")
    st.markdown("Ask anything about your uploaded study materials. Grounded answers with exact page citations.")
    
    has_docs = st.session_state.get("vector_db") is not None
    doc_count = len(st.session_state.get("uploaded_file_info", []))
    
    # Document Readiness Banner
    if has_docs:
        st.info(f"🟢 **Ready to assist**: `{doc_count}` document(s) currently indexed. Type your question below.")
    else:
        st.warning("⚠️ **No documents indexed**: Head to **📁 My Documents** to upload your notes before asking questions.")
        
    # Suggested Study Questions (Generic academic examples)
    with st.expander("💡 Example questions you can ask your study materials", expanded=False):
        st.markdown("""
        - *"Can you explain the primary concepts discussed in chapter 1 in simple terms?"*
        - *"What are the key formulas or technical definitions provided in the text?"*
        - *"Provide a concise step-by-step summary of the mechanism described on page 2."*
        - *"Contrast the advantages and disadvantages outlined in the material."*
        """)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render Chat History
    for msg in st.session_state.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        citations = msg.get("citations", [])
        model_used = msg.get("model", "")
        
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🎓"):
                st.markdown(content)
                
                # Render verified source citations
                if citations:
                    with st.expander(f"📚 Sources ({len(citations)} references)", expanded=False):
                        for cit in citations:
                            st.markdown(f"**📄 {cit['source']}** — *Page {cit['page_number']}*")
                            st.caption(f"> {cit['snippet']}")
                            st.markdown("---")
                            
                if model_used:
                    st.caption(f"🤖 Powered by `{model_used}`")
                    
    # Chat Input Box
    user_query = st.chat_input(
        "Ask a question about your study documents...",
        disabled=not has_docs
    )
    
    if user_query:
        # Append user message
        st.session_state["messages"].append({
            "role": "user",
            "content": user_query,
            "citations": []
        })
        
        # Display immediately
        with st.chat_message("user"):
            st.markdown(user_query)
            
        vector_db = st.session_state.get("vector_db")
        
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("🧠 Thinking & searching your documents..."):
                source_docs = retrieve_chunks(vector_db, user_query, top_k=TOP_K)
                
                # Check for insufficient evidence
                if not source_docs:
                    answer_text = INSUFFICIENT_EVIDENCE_MSG
                    st.markdown(answer_text)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": [],
                        "model": "chatturAI Grounding Guard"
                    })
                    log_event("No relevant source chunks found; returned grounding guard message.")
                else:
                    citations = format_citations_metadata(source_docs)
                    prompt = get_rag_prompt_template()
                    
                    groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
                    openai_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
                    
                    models_to_try = []
                    if GROQ_AVAILABLE and groq_key:
                        models_to_try.extend([(m, "groq") for m in GROQ_MODELS])
                    if OPENAI_AVAILABLE and openai_key:
                        models_to_try.append(("gpt-3.5-turbo", "openai"))
                        
                    if not models_to_try:
                        err_text = "❌ No LLM configured. Please provide GROQ_API_KEY in your .env file."
                        st.error(err_text)
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": err_text,
                            "citations": []
                        })
                    else:
                        doc_hashes = [c.metadata.get("doc_hash", "") for c in source_docs]
                        cache_key = compute_study_cache_key("ai_tutor", doc_hashes, user_query)
                        cached_resp = st.session_state.get("feature_cache", {}).get(cache_key)

                        if cached_resp:
                            answer_text = cached_resp["answer"]
                            citations = cached_resp.get("citations", citations)
                            st.markdown(answer_text)
                            if citations:
                                with st.expander(f"📚 Sources ({len(citations)} references)", expanded=True):
                                    for cit in citations:
                                        st.markdown(f"**📄 {cit['source']}** — *Page {cit['page_number']}*")
                                        st.caption(f"> {cit['snippet']}")
                                        st.markdown("---")
                            st.caption(f"🤖 Powered by `{cached_resp.get('model', 'qwen/qwen3.8-27b')}` *(session cache)*")
                            st.session_state["messages"].append({
                                "role": "assistant",
                                "content": answer_text,
                                "citations": citations,
                                "model": f"{cached_resp.get('model', 'qwen/qwen3.8-27b')} (cached)"
                            })
                            sync_current_workspace()
                            log_event("Reused session-cached AI Tutor response")
                        else:
                            answered = False
                            for model_name, provider in models_to_try:
                                if provider == "groq":
                                    llm = get_groq_llm_instance(groq_key, model_name, max_tokens=TOKEN_BUDGETS["ai_tutor"])
                                else:
                                    llm = OpenAI(temperature=0.0, openai_api_key=openai_key)
                                    
                                doc_chain = create_stuff_documents_chain(llm, prompt)
                                answer_text, err_msg = invoke_llm_resilient(
                                    doc_chain,
                                    {"context": source_docs, "input": user_query},
                                    feature_name="ai_tutor"
                                )
                                
                                if err_msg:
                                    log_event(f"Model {model_name} failed: {err_msg}", "WARNING")
                                    continue
                                
                                st.markdown(answer_text)
                                if citations:
                                    with st.expander(f"📚 Sources ({len(citations)} references)", expanded=True):
                                        for cit in citations:
                                            st.markdown(f"**📄 {cit['source']}** — *Page {cit['page_number']}*")
                                            st.caption(f"> {cit['snippet']}")
                                            st.markdown("---")
                                st.caption(f"🤖 Powered by `{model_name}`")
                                
                                st.session_state["messages"].append({
                                    "role": "assistant",
                                    "content": answer_text,
                                    "citations": citations,
                                    "model": model_name
                                })
                                st.session_state["feature_cache"][cache_key] = {
                                    "answer": answer_text,
                                    "citations": citations,
                                    "model": model_name
                                }
                                sync_current_workspace()
                                log_event(f"Generated grounded answer with {model_name}")
                                answered = True
                                break
                                    
                            if not answered:
                                st.error("❌ Unable to generate response with available models. Please try again.")
                                st.session_state["messages"].append({
                                    "role": "assistant",
                                    "content": "❌ Unable to generate response. Please check your API key and connection.",
                                    "citations": []
                                })
                                sync_current_workspace()
                            
    # Chat Actions Toolbar
    if st.session_state.get("messages"):
        st.markdown("<br>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state["messages"] = []
                sync_current_workspace()
                st.rerun()
        with col_c2:
            chat_export = f"chatturAI Study Session Export — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            chat_export += "=" * 60 + "\n\n"
            for m in st.session_state["messages"]:
                chat_export += f"[{m['role'].upper()}]: {m['content']}\n"
                if m.get("citations"):
                    chat_export += "Sources:\n"
                    for c in m["citations"]:
                        chat_export += f" - {c['source']} (Page {c['page_number']})\n"
                chat_export += "\n" + "-" * 40 + "\n\n"
            st.download_button(
                "📥 Export Conversation (.txt)",
                data=chat_export,
                file_name=f"chatturAI_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )


# ==============================================================================
# SECTION D: SMART NOTES & EXPLAIN SIMPLY
# ==============================================================================
elif st.session_state["nav_section"] == "📝 Smart Notes":
    st.markdown("## 📝 Smart Notes & Conceptual Explanations")
    st.markdown("Synthesize uploaded materials into structured notes or break down complex concepts with intuitive analogies.")
    
    has_docs = (st.session_state.get("vector_db") is not None) and bool(st.session_state.get("uploaded_file_info"))
    
    if not has_docs:
        st.warning("⚠️ **No study materials indexed**: Head to **📁 My Documents** to upload your notes before generating Smart Notes.")
    else:
        all_doc_names = [d["name"] for d in st.session_state.get("uploaded_file_info", [])]
        
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            selected_docs = st.multiselect(
                "Select Notes Scope (one or more PDFs):",
                options=all_doc_names,
                default=all_doc_names,
                key="notes_doc_multiselect"
            )
        with col_ctrl2:
            summary_length = st.radio(
                "Summary Depth:",
                options=["Short", "Detailed"],
                horizontal=True,
                key="notes_length_radio"
            )
            
        # Invalidate cached notes if document scope changes
        current_notes_key = ",".join(sorted(selected_docs))
        if st.session_state.get("notes_doc_key") != current_notes_key:
            st.session_state["smart_notes"] = None
            st.session_state["notes_doc_key"] = current_notes_key

        if st.button("📝 Generate Smart Notes", use_container_width=True, disabled=not selected_docs):
            budget = TOKEN_BUDGETS["smart_notes_detailed"] if summary_length == "Detailed" else TOKEN_BUDGETS["smart_notes_short"]
            llm = get_default_study_llm(max_tokens=budget)
            if not llm:
                st.error("❌ No LLM configured. Please provide GROQ_API_KEY in your .env file.")
            else:
                filtered_chunks = [
                    c for c in st.session_state.get("document_chunks", [])
                    if c.metadata.get("source") in selected_docs
                ]
                if not filtered_chunks:
                    # Fallback to docstore chunks if document_chunks was empty
                    vdb = st.session_state.get("vector_db")
                    if vdb and hasattr(vdb, "docstore") and hasattr(vdb.docstore, "_dict"):
                        filtered_chunks = [
                            doc for doc in vdb.docstore._dict.values()
                            if doc.metadata.get("source") in selected_docs
                        ]
                
                if not filtered_chunks:
                    st.error("❌ No document passages found for the selected scope.")
                else:
                    doc_fingerprints = [c.metadata.get("doc_hash", "") for c in filtered_chunks]
                    cache_key = compute_study_cache_key("smart_notes", doc_fingerprints, summary_length)
                    cached_notes = st.session_state.get("feature_cache", {}).get(cache_key)

                    if cached_notes and not cached_notes.get("error"):
                        st.session_state["smart_notes"] = cached_notes
                        sync_current_workspace()
                        log_event(f"Reused session-cached {summary_length} Smart Notes")
                        st.rerun()
                    else:
                        with st.spinner("🧠 Synthesizing structured study notes from your documents..."):
                            res = generate_smart_notes(filtered_chunks, summary_length, llm)
                            if res.get("error"):
                                st.error(f"❌ {res['error']}")
                            else:
                                st.session_state["smart_notes"] = res
                                st.session_state["feature_cache"][cache_key] = res
                                sync_current_workspace()
                                log_event(f"Generated {summary_length} Smart Notes across {len(selected_docs)} document(s)")
                                st.rerun()

        # Display Generated Smart Notes
        if st.session_state.get("smart_notes"):
            notes_data = st.session_state["smart_notes"]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='chattur-card' style='border-left: 4px solid #8b5cf6;'>
            """, unsafe_allow_html=True)
            st.markdown(notes_data["notes"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            citations = notes_data.get("citations", [])
            if citations:
                with st.expander(f"📚 Sources ({len(citations)} references)", expanded=False):
                    for cit in citations:
                        st.markdown(f"**📄 {cit['source']}** — *Page {cit['page_number']}*")
                        st.caption(f"> {cit['snippet']}")
                        st.markdown("---")

        # Feature 2: Explain Simply
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### 💡 Explain Simply")
        st.markdown("Enter any difficult term, formula, or concept from your notes for a simplified breakdown with relatable analogies.")
        
        col_ex1, col_ex2 = st.columns([3, 1])
        with col_ex1:
            concept_input = st.text_input(
                "Concept to break down:",
                placeholder="e.g. Backpropagation, Entropy, Raft consensus algorithm",
                key="explain_concept_input"
            )
        with col_ex2:
            st.write("")
            st.write("")
            explain_btn = st.button("✨ Explain Concept", use_container_width=True, disabled=not concept_input.strip())
            
        if explain_btn and concept_input.strip():
            llm = get_default_study_llm(max_tokens=TOKEN_BUDGETS["explain_simply"])
            if not llm:
                st.error("❌ No LLM configured. Please check GROQ_API_KEY in .env.")
            else:
                doc_fingerprints = [c.metadata.get("doc_hash", "") for c in st.session_state.get("document_chunks", [])]
                cache_key = compute_study_cache_key("explain_simply", doc_fingerprints, concept_input.strip())
                cached_res = st.session_state.get("feature_cache", {}).get(cache_key)

                if cached_res and not cached_res.get("error"):
                    st.session_state["explain_result"] = cached_res
                    sync_current_workspace()
                    log_event(f"Reused session-cached Explain Simply for: {concept_input}")
                    st.rerun()
                else:
                    with st.spinner(f"🔍 Breaking down '{concept_input}' simply..."):
                        res = explain_simply(concept_input, st.session_state.get("vector_db"), llm)
                        if not res.get("error"):
                            st.session_state["feature_cache"][cache_key] = res
                        st.session_state["explain_result"] = res
                        sync_current_workspace()
                        log_event(f"Generated Explain Simply for: {concept_input}")
                        st.rerun()

        if st.session_state.get("explain_result"):
            ex_res = st.session_state["explain_result"]
            if ex_res.get("error"):
                st.error(f"❌ {ex_res['error']}")
            else:
                st.markdown("""
                <div class='chattur-card' style='border-left: 4px solid #c084fc; margin-top: 1rem;'>
                """, unsafe_allow_html=True)
                st.markdown(ex_res["explanation"])
                st.markdown("</div>", unsafe_allow_html=True)
                
                ex_citations = ex_res.get("citations", [])
                if ex_citations:
                    with st.expander(f"📚 Sources ({len(ex_citations)} references)", expanded=False):
                        for cit in ex_citations:
                            st.markdown(f"**📄 {cit['source']}** — *Page {cit['page_number']}*")
                            st.caption(f"> {cit['snippet']}")
                            st.markdown("---")


# ==============================================================================
# SECTION E: PRACTICE (MCQ QUIZ & FLASHCARDS)
# ==============================================================================
elif st.session_state["nav_section"] == "⚡ Practice":
    st.markdown("## ⚡ Practice & Active Recall")
    st.markdown("Test your true conceptual understanding with adaptive quizzes and interactive revision flashcards.")
    
    has_docs = (st.session_state.get("vector_db") is not None) and bool(st.session_state.get("uploaded_file_info"))
    
    if not has_docs:
        st.warning("⚠️ **No study materials indexed**: Head to **📁 My Documents** to upload your notes before practicing.")
    else:
        tab_quiz, tab_flashcards = st.tabs(["🎯 MCQ Quiz", "🎴 Flashcards"])
        all_doc_names = [d["name"] for d in st.session_state.get("uploaded_file_info", [])]

        # ----------------------------------------------------------------------
        # SUB-TAB 1: MCQ QUIZ
        # ----------------------------------------------------------------------
        with tab_quiz:
            st.markdown("### 🎯 Concept Mastery Quiz")
            st.caption("5 multiple-choice questions grounded strictly in your uploaded course materials.")
            
            quiz_scope = st.multiselect(
                "Select Quiz Scope:",
                options=all_doc_names,
                default=all_doc_names,
                key="quiz_scope_multiselect"
            )
            
            # Detect document scope change to invalidate stale quiz
            quiz_key = ",".join(sorted(quiz_scope))
            if st.session_state.get("quiz_doc_key") != quiz_key:
                st.session_state["current_quiz"] = None
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_score"] = None
                st.session_state["quiz_doc_key"] = quiz_key
                
            col_q_act1, col_q_act2 = st.columns([2, 1])
            with col_q_act1:
                gen_quiz_btn = st.button("🎲 Generate 5-Question Quiz", use_container_width=True, disabled=not quiz_scope)
                
            if gen_quiz_btn and quiz_scope:
                llm = get_default_study_llm(max_tokens=TOKEN_BUDGETS["mcq_quiz"])
                if not llm:
                    st.error("❌ No LLM configured. Please check GROQ_API_KEY in .env.")
                else:
                    filtered_chunks = [
                        c for c in st.session_state.get("document_chunks", [])
                        if c.metadata.get("source") in quiz_scope
                    ]
                    if not filtered_chunks:
                        vdb = st.session_state.get("vector_db")
                        if vdb and hasattr(vdb, "docstore") and hasattr(vdb.docstore, "_dict"):
                            filtered_chunks = [
                                doc for doc in vdb.docstore._dict.values()
                                if doc.metadata.get("source") in quiz_scope
                            ]
                            
                    if not filtered_chunks:
                        st.error("❌ No passages available for the selected scope.")
                    else:
                        doc_fingerprints = [c.metadata.get("doc_hash", "") for c in filtered_chunks]
                        cache_key = compute_study_cache_key("mcq_quiz", doc_fingerprints, "mcq_5")
                        cached_quiz = st.session_state.get("feature_cache", {}).get(cache_key)

                        if cached_quiz and len(cached_quiz) == 5:
                            st.session_state["current_quiz"] = cached_quiz
                            st.session_state["quiz_answers"] = {}
                            st.session_state["quiz_submitted"] = False
                            st.session_state["quiz_score"] = None
                            sync_current_workspace()
                            log_event(f"Reused session-cached 5-question quiz across {len(quiz_scope)} document(s)")
                            st.rerun()
                        else:
                            with st.spinner("🎲 Crafting 5 challenging MCQs with strict grounding..."):
                                res = generate_mcq_quiz(filtered_chunks, llm)
                                if res.get("error"):
                                    st.error(f"❌ {res['error']}")
                                else:
                                    st.session_state["current_quiz"] = res["quiz"]
                                    st.session_state["feature_cache"][cache_key] = res["quiz"]
                                    st.session_state["quiz_answers"] = {}
                                    st.session_state["quiz_submitted"] = False
                                    st.session_state["quiz_score"] = None
                                    sync_current_workspace()
                                    log_event(f"Generated 5-question quiz across {len(quiz_scope)} document(s)")
                                    st.rerun()

            # Render Quiz Questions
            current_quiz = st.session_state.get("current_quiz")
            if current_quiz:
                st.markdown("<br>", unsafe_allow_html=True)
                is_submitted = st.session_state.get("quiz_submitted", False)
                
                # Show score card if submitted
                if is_submitted and st.session_state.get("quiz_score") is not None:
                    score = st.session_state["quiz_score"]
                    pct = int((score / len(current_quiz)) * 100)
                    accent_color = "#34d399" if score >= 4 else ("#fbbf24" if score >= 3 else "#f87171")
                    st.markdown(f"""
                    <div class='chattur-card' style='text-align: center; border: 2px solid {accent_color}; margin-bottom: 1.5rem;'>
                        <h3 style='margin: 0; color: {accent_color};'>Quiz Results</h3>
                        <div style='font-size: 2.8rem; font-weight: 800; color: #f8fafc; margin: 0.5rem 0;'>{score} / {len(current_quiz)} <span style='font-size: 1.4rem; color: #94a3b8;'>({pct}%)</span></div>
                        <p style='color: #cbd5e1; margin: 0;'>{"🎉 Excellent mastery! Ratta Nahi, Samajh Ke Padha!" if score >= 4 else "Review the explanations below to strengthen your understanding."}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Render each question
                for q_idx, q in enumerate(current_quiz):
                    st.markdown(f"#### **Q{q_idx + 1}. {q['question']}**")
                    options = q["options"]
                    user_selection = st.session_state["quiz_answers"].get(q_idx)
                    
                    if not is_submitted:
                        # Radio choice before submission
                        chosen_opt = st.radio(
                            f"Question {q_idx + 1} Options",
                            options=options,
                            index=user_selection,
                            key=f"quiz_radio_{q_idx}",
                            label_visibility="collapsed"
                        )
                        if chosen_opt in options:
                            st.session_state["quiz_answers"][q_idx] = options.index(chosen_opt)
                    else:
                        # Display evaluation after submission
                        correct_idx = q["correct_index"]
                        is_correct = (user_selection == correct_idx)
                        
                        for opt_idx, opt_text in enumerate(options):
                            if opt_idx == correct_idx:
                                st.markdown(f"🟢 **Option {opt_idx + 1}:** {opt_text} *(Correct Answer)*")
                            elif opt_idx == user_selection and not is_correct:
                                st.markdown(f"🔴 **Option {opt_idx + 1}:** {opt_text} *(Your Answer)*")
                            else:
                                st.markdown(f"⚪ Option {opt_idx + 1}: {opt_text}")
                                
                        if is_correct:
                            st.success(f"✅ Correct! (Page {q.get('source_page', 1)})")
                        else:
                            st.error(f"❌ Incorrect. (Page {q.get('source_page', 1)})")
                            
                        st.info(f"💡 **Explanation:** {q['explanation']}")
                        
                    st.markdown("---")

                # Submission and Reset Controls
                col_sub1, col_sub2 = st.columns([2, 1])
                if not is_submitted:
                    with col_sub1:
                        if st.button("📤 Submit Quiz", use_container_width=True):
                            # Calculate score purely locally in Python
                            local_score = sum(
                                1 for i, q in enumerate(current_quiz)
                                if st.session_state["quiz_answers"].get(i) == q["correct_index"]
                            )
                            st.session_state["quiz_submitted"] = True
                            st.session_state["quiz_score"] = local_score
                            sync_current_workspace()
                            log_event(f"Quiz submitted. Score: {local_score}/{len(current_quiz)}")
                            st.rerun()
                else:
                    with col_sub1:
                        if st.button("🔄 Retake / New Quiz", use_container_width=True):
                            st.session_state["current_quiz"] = None
                            st.session_state["quiz_answers"] = {}
                            st.session_state["quiz_submitted"] = False
                            st.session_state["quiz_score"] = None
                            sync_current_workspace()
                            st.rerun()

        # ----------------------------------------------------------------------
        # SUB-TAB 2: FLASHCARDS
        # ----------------------------------------------------------------------
        with tab_flashcards:
            st.markdown("### 🎴 Active Recall Flashcards")
            st.caption("Flip through high-yield revision cards generated directly from your documents.")
            
            fc_scope = st.multiselect(
                "Select Flashcard Scope:",
                options=all_doc_names,
                default=all_doc_names,
                key="fc_scope_multiselect"
            )
            
            # Detect document scope change to invalidate stale flashcards
            fc_key = ",".join(sorted(fc_scope))
            if st.session_state.get("flashcards_doc_key") != fc_key:
                st.session_state["flashcards"] = None
                st.session_state["flashcards_doc_key"] = fc_key
                st.session_state["flashcard_idx"] = 0
                st.session_state["flashcard_show_back"] = False
                
            if st.button("🎴 Generate Flashcard Deck", use_container_width=True, disabled=not fc_scope):
                llm = get_default_study_llm(max_tokens=TOKEN_BUDGETS["flashcards"])
                if not llm:
                    st.error("❌ No LLM configured. Please check GROQ_API_KEY in .env.")
                else:
                    filtered_chunks = [
                        c for c in st.session_state.get("document_chunks", [])
                        if c.metadata.get("source") in fc_scope
                    ]
                    if not filtered_chunks:
                        vdb = st.session_state.get("vector_db")
                        if vdb and hasattr(vdb, "docstore") and hasattr(vdb.docstore, "_dict"):
                            filtered_chunks = [
                                doc for doc in vdb.docstore._dict.values()
                                if doc.metadata.get("source") in fc_scope
                            ]
                            
                    if not filtered_chunks:
                        st.error("❌ No passages available for the selected scope.")
                    else:
                        doc_fingerprints = [c.metadata.get("doc_hash", "") for c in filtered_chunks]
                        cache_key = compute_study_cache_key("flashcards", doc_fingerprints, "flashcards_10")
                        cached_cards = st.session_state.get("feature_cache", {}).get(cache_key)

                        if cached_cards and len(cached_cards) > 0:
                            st.session_state["flashcards"] = cached_cards
                            st.session_state["flashcard_idx"] = 0
                            st.session_state["flashcard_show_back"] = False
                            sync_current_workspace()
                            log_event(f"Reused session-cached {len(cached_cards)} flashcards across {len(fc_scope)} document(s)")
                            st.rerun()
                        else:
                            with st.spinner("🎴 Generating high-yield revision flashcards..."):
                                res = generate_flashcards(filtered_chunks, llm)
                                if res.get("error"):
                                    st.error(f"❌ {res['error']}")
                                else:
                                    st.session_state["flashcards"] = res["flashcards"]
                                    st.session_state["feature_cache"][cache_key] = res["flashcards"]
                                    st.session_state["flashcard_idx"] = 0
                                    st.session_state["flashcard_show_back"] = False
                                    sync_current_workspace()
                                    log_event(f"Generated {len(res['flashcards'])} flashcards across {len(fc_scope)} document(s)")
                                    st.rerun()

            # Render Flashcard Viewer
            flashcards = st.session_state.get("flashcards")
            if flashcards:
                cur_idx = st.session_state.get("flashcard_idx", 0)
                # Ensure index bounds
                cur_idx = max(0, min(cur_idx, len(flashcards) - 1))
                st.session_state["flashcard_idx"] = cur_idx
                card = flashcards[cur_idx]
                show_back = st.session_state.get("flashcard_show_back", False)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                    <span style='color: #a78bfa; font-weight: 700; font-size: 0.9rem;'>DECK REVISION</span>
                    <span style='color: #94a3b8; font-weight: 600;'>Card {cur_idx + 1} of {len(flashcards)}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Flip Card UI Container
                card_title = "📖 ANSWER / EXPLANATION" if show_back else "❓ CONCEPT / QUESTION"
                card_content = card["back"] if show_back else card["front"]
                card_border = "#8b5cf6" if show_back else "#38bdf8"
                card_bg = "rgba(139, 92, 246, 0.08)" if show_back else "rgba(56, 189, 248, 0.06)"
                
                st.markdown(f"""
                <div class='chattur-card' style='min-height: 180px; display: flex; flex-direction: column; justify-content: center; text-align: center; border: 2px solid {card_border}; background: {card_bg}; padding: 2rem; border-radius: 16px;'>
                    <div style='font-size: 0.8rem; font-weight: 700; color: {card_border}; letter-spacing: 1px; margin-bottom: 0.8rem;'>
                        {card_title}
                    </div>
                    <div style='font-size: 1.25rem; font-weight: 600; color: #f8fafc; line-height: 1.6;'>
                        {card_content}
                    </div>
                    {f"<div style='margin-top: 1rem; color: #94a3b8; font-size: 0.85rem;'>📄 Source Reference: Page {card.get('page', 1)}</div>" if show_back else ""}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Flashcard Navigation Controls
                col_c_prev, col_c_flip, col_c_next = st.columns([1, 2, 1])
                with col_c_prev:
                    if st.button("⬅️ Previous", use_container_width=True, disabled=cur_idx <= 0):
                        st.session_state["flashcard_idx"] = max(0, cur_idx - 1)
                        st.session_state["flashcard_show_back"] = False
                        sync_current_workspace()
                        st.rerun()
                with col_c_flip:
                    toggle_text = "🙈 Hide Answer" if show_back else "👁️ Reveal Answer"
                    if st.button(toggle_text, use_container_width=True):
                        st.session_state["flashcard_show_back"] = not show_back
                        sync_current_workspace()
                        st.rerun()
                with col_c_next:
                    if st.button("Next ➡️", use_container_width=True, disabled=cur_idx >= len(flashcards) - 1):
                        st.session_state["flashcard_idx"] = min(len(flashcards) - 1, cur_idx + 1)
                        st.session_state["flashcard_show_back"] = False
                        sync_current_workspace()
                        st.rerun()

