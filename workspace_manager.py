"""
chatturAI Workspace Manager — Advanced Phase 3
Implements lightweight, reliable local workspace persistence for study sessions,
documents, FAISS vector indexes, and AI study features.
"""

import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DEFAULT_STORAGE_ROOT = Path(__file__).parent / "workspaces_storage"
WORKSPACE_ID_REGEX = re.compile(r"^ws_[a-zA-Z0-9_-]{4,32}$")


class WorkspaceManager:
    """Manages persistent SQLite metadata and local storage for student workspaces."""

    def __init__(self, storage_root: Optional[Path] = None):
        self.storage_root = Path(storage_root or DEFAULT_STORAGE_ROOT).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_root / "workspaces.db"
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_database(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    doc_fingerprint TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS workspace_documents (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    doc_hash TEXT NOT NULL,
                    size_kb REAL NOT NULL,
                    page_count INTEGER NOT NULL,
                    extraction_method TEXT DEFAULT 'text',
                    chunks_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS workspace_study_state (
                    workspace_id TEXT PRIMARY KEY,
                    messages_json TEXT DEFAULT '[]',
                    notes_json TEXT DEFAULT '{}',
                    explain_json TEXT DEFAULT '{}',
                    quiz_json TEXT DEFAULT '[]',
                    quiz_state_json TEXT DEFAULT '{}',
                    flashcards_json TEXT DEFAULT '[]',
                    flashcard_idx INTEGER DEFAULT 0,
                    feature_cache_json TEXT DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
                );
            """)

    def _validate_workspace_id(self, workspace_id: str) -> None:
        if not workspace_id or not WORKSPACE_ID_REGEX.match(workspace_id):
            raise ValueError(f"Invalid workspace ID format: '{workspace_id}'")
        target_dir = (self.storage_root / workspace_id).resolve()
        if not str(target_dir).startswith(str(self.storage_root)):
            raise ValueError("Path traversal attempt detected in workspace path.")

    def _get_workspace_dir(self, workspace_id: str) -> Path:
        self._validate_workspace_id(workspace_id)
        ws_dir = self.storage_root / workspace_id
        ws_dir.mkdir(parents=True, exist_ok=True)
        return ws_dir

    def create_workspace(self, name: str) -> Dict[str, Any]:
        """Create a new workspace with a unique ID and initialized storage."""
        clean_name = name.strip() or "Untitled Workspace"
        ws_id = f"ws_{uuid.uuid4().hex[:8]}"
        now = time.time()

        ws_dir = self._get_workspace_dir(ws_id)
        (ws_dir / "faiss").mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO workspaces (id, name, created_at, updated_at, doc_fingerprint) VALUES (?, ?, ?, ?, ?)",
                (ws_id, clean_name, now, now, ""),
            )
            conn.execute(
                "INSERT INTO workspace_study_state (workspace_id, updated_at) VALUES (?, ?)",
                (ws_id, now),
            )

        return {
            "id": ws_id,
            "name": clean_name,
            "created_at": now,
            "updated_at": now,
            "doc_count": 0,
        }

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """List all saved workspaces sorted by last updated."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT w.id, w.name, w.created_at, w.updated_at,
                       COUNT(d.id) AS doc_count
                FROM workspaces w
                LEFT JOIN workspace_documents d ON w.id = d.workspace_id
                GROUP BY w.id
                ORDER BY w.updated_at DESC
            """).fetchall()

        workspaces = []
        for r in rows:
            workspaces.append({
                "id": r["id"],
                "name": r["name"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "doc_count": r["doc_count"],
            })
        return workspaces

    def get_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific workspace."""
        self._validate_workspace_id(workspace_id)
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, name, created_at, updated_at, doc_fingerprint FROM workspaces WHERE id = ?",
                (workspace_id,),
            ).fetchone()
            if not row:
                return None
            doc_count = conn.execute(
                "SELECT COUNT(*) FROM workspace_documents WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]

        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "doc_fingerprint": row["doc_fingerprint"],
            "doc_count": doc_count,
        }

    def rename_workspace(self, workspace_id: str, new_name: str) -> bool:
        """Rename an existing workspace."""
        self._validate_workspace_id(workspace_id)
        clean_name = new_name.strip()
        if not clean_name:
            return False

        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE workspaces SET name = ?, updated_at = ? WHERE id = ?",
                (clean_name, time.time(), workspace_id),
            )
            return cursor.rowcount > 0

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete workspace metadata from SQLite and remove local files safely."""
        self._validate_workspace_id(workspace_id)
        ws_dir = self.storage_root / workspace_id

        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
            deleted = cursor.rowcount > 0

        if ws_dir.exists():
            shutil.rmtree(ws_dir, ignore_errors=True)

        return deleted

    def save_workspace_state(
        self,
        workspace_id: str,
        uploaded_file_info: List[Dict[str, Any]],
        document_chunks: List[Document],
        vector_db: Optional[FAISS],
        messages: Optional[List[Dict[str, Any]]] = None,
        smart_notes: Optional[Dict[str, Any]] = None,
        explain_result: Optional[Dict[str, Any]] = None,
        current_quiz: Optional[List[Dict[str, Any]]] = None,
        quiz_state: Optional[Dict[str, Any]] = None,
        flashcards: Optional[List[Dict[str, Any]]] = None,
        flashcard_idx: int = 0,
        feature_cache: Optional[Dict[str, Any]] = None,
        files_data: Optional[Dict[str, bytes]] = None,
    ) -> bool:
        """Persist documents, FAISS index, and study state atomically."""
        self._validate_workspace_id(workspace_id)
        ws_dir = self._get_workspace_dir(workspace_id)
        now = time.time()

        # Compute document fingerprint
        doc_hashes = sorted([f.get("hash", f.get("name", "")) for f in uploaded_file_info])
        doc_fingerprint = ",".join(doc_hashes)

        # 1. Save uploaded PDF files to documents/ directory if provided
        if files_data:
            docs_dir = ws_dir / "documents"
            docs_dir.mkdir(parents=True, exist_ok=True)
            for fname, fbytes in files_data.items():
                safe_fname = Path(fname).name
                target_file = docs_dir / safe_fname
                try:
                    target_file.write_bytes(fbytes)
                except Exception:
                    pass

        # 2. Save FAISS index safely if present, or clear if invalidated
        faiss_dir = ws_dir / "faiss"
        if vector_db is not None:
            faiss_tmp = ws_dir / "faiss_tmp"
            if faiss_tmp.exists():
                shutil.rmtree(faiss_tmp, ignore_errors=True)
            faiss_tmp.mkdir(parents=True, exist_ok=True)

            try:
                vector_db.save_local(str(faiss_tmp))
                if faiss_dir.exists():
                    shutil.rmtree(faiss_dir, ignore_errors=True)
                faiss_tmp.rename(faiss_dir)
            except Exception as e:
                shutil.rmtree(faiss_tmp, ignore_errors=True)
                raise RuntimeError(f"Failed to persist FAISS index: {e}")
        else:
            if faiss_dir.exists():
                shutil.rmtree(faiss_dir, ignore_errors=True)

        if not uploaded_file_info:
            docs_dir = ws_dir / "documents"
            if docs_dir.exists():
                shutil.rmtree(docs_dir, ignore_errors=True)

        # Group chunks by document filename
        chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for c in document_chunks:
            src = c.metadata.get("source", "document.pdf")
            if src not in chunks_by_doc:
                chunks_by_doc[src] = []
            chunks_by_doc[src].append({
                "page_content": c.page_content,
                "metadata": c.metadata,
            })

        # 2. Update SQLite records in transaction
        with self._get_connection() as conn:
            conn.execute("DELETE FROM workspace_documents WHERE workspace_id = ?", (workspace_id,))

            for info in uploaded_file_info:
                fname = info["name"]
                d_hash = info.get("hash", info.get("name"))
                sz = info.get("size_kb", 0.0)
                chunks_for_file = chunks_by_doc.get(fname, [])
                pg_count = max(
                    [c["metadata"].get("total_pages", c["metadata"].get("page_number", 1)) for c in chunks_for_file] or [1]
                )
                ext_method = chunks_for_file[0]["metadata"].get("extraction_method", "text") if chunks_for_file else "text"

                conn.execute(
                    """INSERT INTO workspace_documents 
                       (id, workspace_id, filename, doc_hash, size_kb, page_count, extraction_method, chunks_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"{workspace_id}_{d_hash}",
                        workspace_id,
                        fname,
                        d_hash,
                        sz,
                        pg_count,
                        ext_method,
                        json.dumps(chunks_for_file),
                        now,
                    ),
                )

            # Update study state
            conn.execute(
                """INSERT INTO workspace_study_state 
                   (workspace_id, messages_json, notes_json, explain_json, quiz_json, quiz_state_json, flashcards_json, flashcard_idx, feature_cache_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(workspace_id) DO UPDATE SET
                   messages_json = excluded.messages_json,
                   notes_json = excluded.notes_json,
                   explain_json = excluded.explain_json,
                   quiz_json = excluded.quiz_json,
                   quiz_state_json = excluded.quiz_state_json,
                   flashcards_json = excluded.flashcards_json,
                   flashcard_idx = excluded.flashcard_idx,
                   feature_cache_json = excluded.feature_cache_json,
                   updated_at = excluded.updated_at""",
                (
                    workspace_id,
                    json.dumps(messages or []),
                    json.dumps(smart_notes or {}),
                    json.dumps(explain_result or {}),
                    json.dumps(current_quiz or []),
                    json.dumps(quiz_state or {}),
                    json.dumps(flashcards or []),
                    flashcard_idx,
                    json.dumps(feature_cache or {}),
                    now,
                ),
            )

            conn.execute(
                "UPDATE workspaces SET updated_at = ?, doc_fingerprint = ? WHERE id = ?",
                (now, doc_fingerprint, workspace_id),
            )

        return True

    def load_workspace_state(
        self,
        workspace_id: str,
        embeddings: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Reload documents, FAISS index, and study features from persistent storage.
        Restores index in milliseconds without re-embedding or invoking LLMs.
        """
        self._validate_workspace_id(workspace_id)
        ws_dir = self.storage_root / workspace_id
        if not ws_dir.exists():
            return None

        with self._get_connection() as conn:
            ws_row = conn.execute("SELECT id, name FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
            if not ws_row:
                return None

            doc_rows = conn.execute(
                "SELECT filename, doc_hash, size_kb, chunks_json FROM workspace_documents WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchall()

            state_row = conn.execute(
                "SELECT * FROM workspace_study_state WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()

        # Reconstruct documents and chunks
        uploaded_file_info: List[Dict[str, Any]] = []
        document_chunks: List[Document] = []
        processed_file_keys: List[str] = []

        for dr in doc_rows:
            fname = dr["filename"]
            sz = dr["size_kb"]
            d_hash = dr["doc_hash"]
            uploaded_file_info.append({
                "name": fname,
                "size_kb": sz,
                "hash": d_hash,
                "status": "Indexed & Ready ✅",
            })
            processed_file_keys.append(f"{fname}_{int(sz * 1024)}")

            try:
                raw_chunks = json.loads(dr["chunks_json"])
                for rc in raw_chunks:
                    document_chunks.append(
                        Document(page_content=rc["page_content"], metadata=rc["metadata"])
                    )
            except Exception:
                pass

        # Reload FAISS index
        vector_db: Optional[FAISS] = None
        faiss_dir = ws_dir / "faiss"
        if (faiss_dir / "index.faiss").exists() and (faiss_dir / "index.pkl").exists():
            try:
                vector_db = FAISS.load_local(
                    str(faiss_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                # Controlled recovery: rebuild from chunks if index file is stale/corrupted
                if document_chunks:
                    vector_db = FAISS.from_documents(document_chunks, embeddings)

        # Parse study state
        messages: List[Dict[str, Any]] = []
        smart_notes: Optional[Dict[str, Any]] = None
        explain_result: Optional[Dict[str, Any]] = None
        current_quiz: Optional[List[Dict[str, Any]]] = None
        quiz_state: Dict[str, Any] = {}
        flashcards: Optional[List[Dict[str, Any]]] = None
        flashcard_idx: int = 0
        feature_cache: Dict[str, Any] = {}

        if state_row:
            try:
                messages = json.loads(state_row["messages_json"] or "[]")
                notes_raw = json.loads(state_row["notes_json"] or "{}")
                smart_notes = notes_raw if notes_raw.get("notes") else None
                explain_raw = json.loads(state_row["explain_json"] or "{}")
                explain_result = explain_raw if explain_raw.get("explanation") else None
                quiz_raw = json.loads(state_row["quiz_json"] or "[]")
                current_quiz = quiz_raw if quiz_raw else None
                quiz_state = json.loads(state_row["quiz_state_json"] or "{}")
                cards_raw = json.loads(state_row["flashcards_json"] or "[]")
                flashcards = cards_raw if cards_raw else None
                flashcard_idx = state_row["flashcard_idx"] or 0
                feature_cache = json.loads(state_row["feature_cache_json"] or "{}")
            except Exception:
                pass

        return {
            "id": ws_row["id"],
            "name": ws_row["name"],
            "uploaded_file_info": uploaded_file_info,
            "document_chunks": document_chunks,
            "processed_file_keys": processed_file_keys,
            "total_chunks": len(document_chunks),
            "vector_db": vector_db,
            "messages": messages,
            "smart_notes": smart_notes,
            "explain_result": explain_result,
            "current_quiz": current_quiz,
            "quiz_answers": quiz_state.get("quiz_answers", {}),
            "quiz_submitted": quiz_state.get("quiz_submitted", False),
            "quiz_score": quiz_state.get("quiz_score"),
            "flashcards": flashcards,
            "flashcard_idx": flashcard_idx,
            "feature_cache": feature_cache,
        }
