"""
chatturAI Study Tools Module — Stage 4
Implements Smart Notes, Explain Simply, MCQ Quiz Generator, and Interactive Flashcards.
Guarantees document grounding, strict JSON validation, and clean error handling.
"""

import hashlib
import json
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rag_engine import (
    retrieve_chunks,
    format_citations_metadata,
    INSUFFICIENT_EVIDENCE_MSG,
    DEFAULT_TOP_K,
)

# ------------------------------------------------------------------------------
# SHARED LLM REQUEST RESILIENCE & BUDGETS (PHASE 2)
# ------------------------------------------------------------------------------

TOKEN_BUDGETS: Dict[str, int] = {
    "ai_tutor": 800,
    "explain_simply": 750,
    "smart_notes_short": 900,
    "smart_notes_detailed": 1500,
    "mcq_quiz": 1200,
    "flashcards": 1000,
}

MAX_RETRIES = 2
MAX_TOTAL_WAIT_SECONDS = 8.0


def classify_llm_error(error: Exception) -> Tuple[str, bool, Optional[float]]:
    """
    Classify an LLM error into a user-friendly message, retryable status, and optional retry delay.
    Returns (user_message, is_retryable, retry_after_seconds).
    """
    err_str = str(error).lower()

    # 1. Authentication / Invalid API Key (Non-retryable)
    if any(k in err_str for k in ["401", "authentication", "unauthorized", "invalid api key", "invalid_api_key"]):
        return ("Invalid Groq API key. Please check your credentials in the settings.", False, None)

    # 2. Rate Limiting (429) (Retryable)
    if "429" in err_str or "rate limit" in err_str or "rate_limit_exceeded" in err_str:
        retry_delay = None
        match = re.search(r"try again in ([\d\.]+)\s*s", err_str)
        if match:
            try:
                retry_delay = float(match.group(1))
            except ValueError:
                pass
        if retry_delay is None and hasattr(error, "response") and error.response is not None:
            headers = getattr(error.response, "headers", {})
            if "retry-after" in headers:
                try:
                    retry_delay = float(headers["retry-after"])
                except (ValueError, TypeError):
                    pass
        return ("Groq rate limit reached. Please wait a moment before trying again.", True, retry_delay)

    # 3. Timeouts (Retryable)
    if "timeout" in err_str or "timed out" in err_str:
        return ("The AI request timed out. Please try asking again.", True, None)

    # 4. Context Window Exceeded (Non-retryable without changing input)
    if any(k in err_str for k in ["context_length_exceeded", "maximum context length", "too many tokens"]):
        return ("The document passage is too large for the model's context window. Please select fewer documents.", False, None)

    # 5. Transient Server Errors (5xx) (Retryable)
    if any(k in err_str for k in ["500", "502", "503", "504", "internal server error", "service unavailable", "overloaded"]):
        return ("The AI service is temporarily experiencing high load. Please retry in a moment.", True, None)

    # 6. Default Fallback
    return (f"AI request failed: {error}", False, None)


def invoke_llm_resilient(
    invokable: Any,
    payload: Dict[str, Any],
    feature_name: str,
    max_retries: int = MAX_RETRIES,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Execute an LLM chain or model with centralized error handling, exponential backoff,
    and Retry-After compliance.
    Returns (raw_text_output, error_message).
    """
    total_waited = 0.0

    for attempt in range(max_retries + 1):
        try:
            res = invokable.invoke(payload)
            text = res.content if hasattr(res, "content") else str(res)
            return text, None
        except Exception as e:
            user_msg, is_retryable, retry_after = classify_llm_error(e)

            if not is_retryable or attempt >= max_retries:
                return None, user_msg

            if retry_after is not None:
                delay = retry_after
            else:
                delay = min(1.0 * (2 ** attempt) + random.uniform(0.1, 0.4), 4.0)

            if total_waited + delay > MAX_TOTAL_WAIT_SECONDS:
                return None, f"Groq is busy. Please try again in {int(delay + 1)} seconds."

            sleep_fn(delay)
            total_waited += delay

    return None, "Unable to complete request after retries."


def compute_study_cache_key(
    feature: str,
    doc_fingerprints: List[str],
    query_or_config: str,
    model_name: str = "qwen/qwen3.8-27b",
    version: str = "v2",
) -> str:
    """Generate a deterministic hash key for session-scoped response reuse."""
    fingerprint_str = ",".join(sorted(doc_fingerprints))
    combined = f"{feature}:{fingerprint_str}:{query_or_config}:{model_name}:{version}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:24]


# ------------------------------------------------------------------------------
# 1. SMART NOTES
# ------------------------------------------------------------------------------

SMART_NOTES_PROMPT = """You are chatturAI, an expert study assistant.
Generate comprehensive, structured study notes strictly grounded in the provided document passages.

Motto: "Ratta Nahi, Samajh Ke Padho!"

Selected Notes Length: {length_instruction}

CRITICAL RULES:
1. Base all notes strictly on the provided Context. Do NOT invent facts or outside theories.
2. Structure your output clearly using this Markdown format:
   # 📌 [Topic / Document Overview]
   ### 💡 Important Concepts
   ### 📖 Key Definitions
   ### 🔍 Core Explanations
   ### ⭐ High-Yield Revision Points
3. Be clear, organized, and student-friendly.

<context>
{context}
</context>

Generate Study Notes:"""


def get_representative_chunks(
    chunks: List[Document],
    max_total_chars: int = 8000,
    max_chunks: int = 10,
) -> List[Document]:
    """
    Sample representative chunks across the document for comprehensive summarization.
    """
    if not chunks:
        return []
    if len(chunks) <= max_chunks:
        selected = chunks
    else:
        # Evenly stride across the document
        step = len(chunks) / max_chunks
        indices = [int(i * step) for i in range(max_chunks)]
        selected = [chunks[i] for i in indices]

    # Bound total character budget
    bounded: List[Document] = []
    total_chars = 0
    for c in selected:
        if total_chars + len(c.page_content) > max_total_chars:
            break
        bounded.append(c)
        total_chars += len(c.page_content)

    return bounded or chunks[:1]


def generate_smart_notes(
    chunks: List[Document],
    length: str,
    llm: Any,
) -> Dict[str, Any]:
    """
    Generate structured study notes from document chunks.
    """
    if not chunks:
        return {
            "notes": "No document passages available to generate notes.",
            "citations": [],
            "error": "No documents provided"
        }

    sample_chunks = get_representative_chunks(chunks)
    context_text = "\n\n---\n\n".join([c.page_content for c in sample_chunks])

    length_inst = (
        "Concise summary focusing strictly on core definitions and top bullet points."
        if length.lower() == "short"
        else "Detailed, comprehensive academic notes covering concepts, detailed explanations, and revision insights."
    )

    prompt = ChatPromptTemplate.from_template(SMART_NOTES_PROMPT)
    chain = prompt | llm

    raw_text, err = invoke_llm_resilient(
        chain,
        {
            "length_instruction": length_inst,
            "context": context_text,
        },
        feature_name=f"smart_notes_{length.lower()}"
    )

    if err:
        return {
            "notes": "",
            "citations": [],
            "error": err
        }

    citations = format_citations_metadata(sample_chunks)
    return {
        "notes": raw_text,
        "citations": citations,
        "error": None
    }


# ------------------------------------------------------------------------------
# 2. EXPLAIN SIMPLY
# ------------------------------------------------------------------------------

EXPLAIN_SIMPLY_PROMPT = """You are chatturAI. Explain the difficult concept clearly for a student who is learning it for the first time.
Motto: "Ratta Nahi, Samajh Ke Padho!"

CRITICAL RULES:
1. Base your explanation strictly on the facts in the Context. If the context does not contain enough evidence, state:
"{insufficient_evidence_msg}"
2. Format your response exactly as follows:
### 💡 What It Means
[1-2 clear, direct definition sentences]

### 🔍 Simple Explanation
[Step-by-step breakdown using plain, accessible language]

### 🎯 Intuitive Example
[An everyday relatable analogy or concrete example illustrating the concept]

### ⭐ Key Takeaway
[One memorable, high-impact takeaway for exams]

<context>
{context}
</context>

Concept to Explain: {concept}

Answer:"""


def explain_simply(
    concept: str,
    vectorstore: Any,
    llm: Any,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Generate an intuitive, beginner-friendly explanation for a concept grounded in retrieved notes.
    """
    if not vectorstore or not concept.strip():
        return {
            "explanation": INSUFFICIENT_EVIDENCE_MSG,
            "citations": [],
            "error": None
        }

    retrieved = retrieve_chunks(vectorstore, concept, top_k=top_k)
    if not retrieved:
        return {
            "explanation": INSUFFICIENT_EVIDENCE_MSG,
            "citations": [],
            "error": None
        }

    context_text = "\n\n---\n\n".join([d.page_content for d in retrieved])
    prompt = ChatPromptTemplate.from_template(
        EXPLAIN_SIMPLY_PROMPT.format(
            insufficient_evidence_msg=INSUFFICIENT_EVIDENCE_MSG,
            context="{context}",
            concept="{concept}"
        )
    )
    chain = prompt | llm

    raw_text, err = invoke_llm_resilient(
        chain,
        {"context": context_text, "concept": concept},
        feature_name="explain_simply"
    )

    if err:
        return {
            "explanation": "",
            "citations": [],
            "error": err
        }

    citations = format_citations_metadata(retrieved)
    return {
        "explanation": raw_text,
        "citations": citations,
        "error": None
    }


# ------------------------------------------------------------------------------
# 3. MCQ QUIZ GENERATOR
# ------------------------------------------------------------------------------

QUIZ_PROMPT = """You are chatturAI. Generate exactly 5 challenging multiple-choice questions (MCQs) grounded STRICTLY in the provided text.

CRITICAL RULES:
1. Every question must test a real concept directly present in the context.
2. Return ONLY a valid JSON array of 5 objects. Do NOT output any markdown prose, introductions, or apologies outside the JSON.
3. Each question object MUST have these exact keys:
   - "question": string
   - "options": list of exactly 4 unique strings
   - "correct_index": integer between 0 and 3
   - "explanation": string explaining why the answer is correct
   - "source_page": integer indicating the page number mentioned in context, or 1

JSON Schema Example:
[
  {{
    "question": "What is the primary function of mitochondria?",
    "options": ["ATP generation", "Protein synthesis", "Lipid breakdown", "Cell division"],
    "correct_index": 0,
    "explanation": "Mitochondria generate ATP via oxidative phosphorylation as stated in the text.",
    "source_page": 1
  }}
]

<context>
{context}
</context>

Generate 5 MCQs (JSON only):"""


def extract_json_from_response(text: str) -> Any:
    """Safely extract and parse JSON from an LLM response."""
    cleaned = text.strip()
    # Strip markdown code blocks if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    return json.loads(cleaned)


def validate_quiz_structure(quiz_data: Any) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """
    Validate that quiz structure matches strict Stage 4 requirements:
    - Exactly 4 options per question
    - correct_index in 0..3
    - No duplicate options
    - Non-empty questions
    - No duplicate questions
    """
    if not isinstance(quiz_data, list):
        return False, [], "Quiz output is not a JSON array"

    validated: List[Dict[str, Any]] = []
    seen_questions = set()

    for i, item in enumerate(quiz_data):
        if not isinstance(item, dict):
            return False, [], f"Question #{i+1} is not a valid JSON object"

        q_text = str(item.get("question", "")).strip()
        options = item.get("options", [])
        correct_idx = item.get("correct_index")
        explanation = str(item.get("explanation", "")).strip()
        source_page = item.get("source_page", 1)

        if not q_text or len(q_text) < 5:
            return False, [], f"Question #{i+1} has an empty or invalid question text"

        if q_text.lower() in seen_questions:
            return False, [], f"Duplicate question detected: '{q_text}'"
        seen_questions.add(q_text.lower())

        if not isinstance(options, list) or len(options) != 4:
            return False, [], f"Question #{i+1} must have exactly 4 options, found {len(options) if isinstance(options, list) else 0}"

        clean_options = [str(opt).strip() for opt in options]
        if any(not opt for opt in clean_options):
            return False, [], f"Question #{i+1} contains empty options"

        if len(set(opt.lower() for opt in clean_options)) != 4:
            return False, [], f"Question #{i+1} has duplicate options"

        if not isinstance(correct_idx, int) or correct_idx < 0 or correct_idx > 3:
            return False, [], f"Question #{i+1} has invalid correct_index '{correct_idx}' (must be 0, 1, 2, or 3)"

        validated.append({
            "id": i + 1,
            "question": q_text,
            "options": clean_options,
            "correct_index": correct_idx,
            "explanation": explanation or "Grounded in document context.",
            "source_page": int(source_page) if isinstance(source_page, (int, float)) else 1
        })

    if len(validated) < 5:
        return False, [], f"Expected at least 5 questions, found {len(validated)}"

    return True, validated[:5], None


def generate_mcq_quiz(
    chunks: List[Document],
    llm: Any,
) -> Dict[str, Any]:
    """
    Generate 5 validated multiple-choice questions from document chunks with one retry on failure.
    """
    if not chunks:
        return {"quiz": [], "error": "No document chunks available for quiz generation"}

    sample_chunks = get_representative_chunks(chunks)
    context_text = "\n\n---\n\n".join([f"[Page {c.metadata.get('page_number', 1)}] {c.page_content}" for c in sample_chunks])

    prompt = ChatPromptTemplate.from_template(QUIZ_PROMPT)
    chain = prompt | llm

    for attempt in range(2):  # Max 1 retry on malformed JSON
        raw_text, err = invoke_llm_resilient(
            chain,
            {"context": context_text},
            feature_name="mcq_quiz"
        )
        if err:
            return {"quiz": [], "error": err}
        try:
            parsed = extract_json_from_response(raw_text)
            valid, questions, err_msg = validate_quiz_structure(parsed)
            if valid:
                return {"quiz": questions, "error": None}
            if attempt == 1:
                return {"quiz": [], "error": f"Quiz validation failed: {err_msg}"}
        except Exception as e:
            if attempt == 1:
                return {"quiz": [], "error": f"Failed to generate quiz: {e}"}

    return {"quiz": [], "error": "Unable to generate valid quiz structure"}


# ------------------------------------------------------------------------------
# 4. INTERACTIVE FLASHCARDS
# ------------------------------------------------------------------------------

FLASHCARD_PROMPT = """You are chatturAI. Generate up to 10 high-yield study flashcards strictly from the provided text.

CRITICAL RULES:
1. Every card must test a core concept, definition, formula, or relationship directly in the text.
2. Return ONLY a valid JSON array of objects. Do NOT include markdown introduction or summary outside the JSON.
3. Each flashcard object MUST have these exact keys:
   - "front": string (question or key term)
   - "back": string (concise answer or definition)
   - "page": integer (page number in text or 1)

JSON Schema Example:
[
  {{
    "front": "What is the Second Law of Thermodynamics?",
    "back": "Total entropy of an isolated system always increases over time.",
    "page": 2
  }}
]

<context>
{context}
</context>

Generate Flashcards (JSON only):"""


def validate_flashcards(data: Any) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """Validate flashcard structure and non-empty fronts/backs."""
    if not isinstance(data, list):
        return False, [], "Flashcard output is not a JSON list"

    cards: List[Dict[str, Any]] = []
    seen = set()

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()
        page = item.get("page", 1)

        if not front or not back:
            continue

        if front.lower() in seen:
            continue
        seen.add(front.lower())

        cards.append({
            "id": len(cards) + 1,
            "front": front,
            "back": back,
            "page": int(page) if isinstance(page, (int, float)) else 1
        })

    if not cards:
        return False, [], "No valid flashcards found in output"

    return True, cards[:10], None


def generate_flashcards(
    chunks: List[Document],
    llm: Any,
) -> Dict[str, Any]:
    """
    Generate up to 10 interactive revision flashcards from document chunks.
    """
    if not chunks:
        return {"flashcards": [], "error": "No document chunks available for flashcard generation"}

    sample_chunks = get_representative_chunks(chunks)
    context_text = "\n\n---\n\n".join([f"[Page {c.metadata.get('page_number', 1)}] {c.page_content}" for c in sample_chunks])

    prompt = ChatPromptTemplate.from_template(FLASHCARD_PROMPT)
    chain = prompt | llm

    raw_text, err = invoke_llm_resilient(
        chain,
        {"context": context_text},
        feature_name="flashcards"
    )
    if err:
        return {"flashcards": [], "error": err}

    try:
        parsed = extract_json_from_response(raw_text)
        valid, cards, err_msg = validate_flashcards(parsed)
        if valid:
            return {"flashcards": cards, "error": None}
        return {"flashcards": [], "error": f"Flashcard validation failed: {err_msg}"}
    except Exception as e:
        return {"flashcards": [], "error": f"Failed to generate flashcards: {e}"}
