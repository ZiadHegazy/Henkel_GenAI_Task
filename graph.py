"""
graph.py - LangGraph RAG State Machine with Intent Routing & Provenance
========================================================================

Architecture Overview:
----------------------
1. Intent Prediction:
   - Classifies query intent ('greeting' vs. 'document_query').
   - Handles conversational greetings ("hello", "who are you") warmly without
     failing into document rejection.

2. State Management & Multi-Turn Memory:
   - Maintains a TypedDict state (`GraphState`) holding conversation messages,
     user query, intent, retrieved documents, similarity scores, and structured answers.
   - Leverages conversation history across turns to resolve pronouns and follow-up queries.

3. Strict Provenance Output Schema:
   - Uses Pydantic `CitedAnswer` to enforce answerability, synthesized text,
     1-to-1 page citations (`source_pages`), and section citations (`source_sections`).

4. Retrieval & Semantic Routing:
   - Fetches candidate chunks from Qdrant Cloud.
   - Enforces the 0.72 Cosine Similarity Gate: if max(similarity_scores) < 0.72,
     routes to the Rejection Node.

5. Prompt Grounding & Model Fallbacks:
   - Injects section and page tags directly into chunk texts inside `<document_context>`.
   - Uses Groq (`llama-3.3-70b-versatile`) as primary generator bound with structured output.
   - Configures Google (`gemini-2.5-flash`) via `.with_fallbacks()` for resilient failover.
"""

from __future__ import annotations

import os
import time
from typing import Annotated, Any, Dict, List, Optional, Sequence
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from logger import get_json_logger, log_generation_event, log_retrieval_event

# Ensure environment variables are loaded
load_dotenv()

rag_logger = get_json_logger("rag_graph")

REJECTION_MESSAGE = "I cannot find the answer to this question in the document."
SIMILARITY_THRESHOLD = 0.72

GREETING_MESSAGE = (
    "Hello! 👋 I am your Enterprise Document Assistant.\n\n"
    "I am specialized in answering questions strictly grounded in your indexed documents, "
    "complete with verified 1-to-1 page citations and hierarchical section tracking.\n\n"
    "**You can ask me about:**\n"
    "- Key facts, methodologies, metrics, and findings in your documents\n"
    "- Specific data points, clauses, or policy details\n"
    "- Section summaries and comparative insights\n\n"
    "What question would you like to explore today?"
)

GREETING_PATTERNS = {
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "howdy", "sup", "yo", "who are you", "what are you",
    "what can you do", "help", "what is this", "thanks", "thank you",
    "bye", "goodbye", "see you", "how are you", "nice to meet you",
    "welcome", "start", "restart", "clear"
}


# ------------------------------------------------------------------------------
# 1. Structured Output Schema & State Definitions
# ------------------------------------------------------------------------------

class CitedAnswer(BaseModel):
    """
    Pydantic output schema enforcing strict grounding and verifiable citations.
    """
    is_answerable: bool = Field(
        description="Set to True if the document context provides sufficient factual evidence to answer the question; False otherwise."
    )
    answer: str = Field(
        description="Factual answer synthesized strictly from the document context. If is_answerable is False, must state: 'I cannot find the answer to this question in the document.'"
    )
    source_pages: List[int] = Field(
        default_factory=list,
        description="List of unique 1-indexed page numbers containing the facts used in the answer."
    )
    source_sections: List[str] = Field(
        default_factory=list,
        description="List of unique section headings containing the facts used in the answer."
    )

    @field_validator("source_pages", mode="before")
    @classmethod
    def sanitize_pages(cls, v: Any) -> List[int]:
        if not v:
            return []
        cleaned: List[int] = []
        for item in v:
            try:
                page_int = int(item)
                if page_int not in cleaned:
                    cleaned.append(page_int)
            except (ValueError, TypeError):
                continue
        return sorted(cleaned)

    @field_validator("source_sections", mode="before")
    @classmethod
    def sanitize_sections(cls, v: Any) -> List[str]:
        if not v:
            return []
        cleaned: List[str] = []
        for item in v:
            sec_str = str(item).strip()
            if sec_str and sec_str not in cleaned:
                cleaned.append(sec_str)
        return cleaned


class GraphState(TypedDict):
    """
    State schema for the LangGraph RAG pipeline.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    intent: Optional[str]
    documents: List[Document]
    similarity_scores: List[float]
    cited_answer: Optional[CitedAnswer]


class PydanticGraphState(BaseModel):
    """
    Complementary Pydantic schema for state validation and testing.
    """
    messages: List[Any] = Field(default_factory=list)
    question: str
    intent: Optional[str] = None
    documents: List[Any] = Field(default_factory=list)
    similarity_scores: List[float] = Field(default_factory=list)
    cited_answer: Optional[CitedAnswer] = None


# ------------------------------------------------------------------------------
# 2. Intent Prediction Logic
# ------------------------------------------------------------------------------

def predict_intent(query: str) -> str:
    """
    Classify query intent into 'greeting' or 'document_query'.
    Prevents casual greetings from triggering document rejection.
    """
    cleaned = query.strip().lower()
    stripped = "".join(ch for ch in cleaned if ch.isalnum() or ch.isspace()).strip()

    if stripped in GREETING_PATTERNS:
        return "greeting"

    for pattern in GREETING_PATTERNS:
        if stripped.startswith(pattern) and len(stripped) <= len(pattern) + 12:
            return "greeting"

    tokens = stripped.split()
    if len(tokens) <= 3 and any(t in ("hello", "hi", "hey", "thanks", "thank", "help") for t in tokens):
        return "greeting"

    return "document_query"


# ------------------------------------------------------------------------------
# 3. Vector Store Initialization
# ------------------------------------------------------------------------------

def get_vector_store(collection_name: Optional[str] = None):
    """Initialize and return QdrantVectorStore connected to Qdrant Cloud."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
    if not gemini_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment variables.")

    os.environ["GOOGLE_API_KEY"] = gemini_key

    # Auto-resolve active Google embedding model with fallback
    candidate_models = [
        os.getenv("EMBEDDING_MODEL"),
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "models/gemini-embedding-2",
        "models/embedding-001",
        "models/text-embedding-004",
    ]
    embeddings = None
    for model_name in [m for m in candidate_models if m]:
        try:
            probe_emb = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=gemini_key)
            _ = probe_emb.embed_query("probe")
            embeddings = probe_emb
            break
        except Exception:
            continue
    if embeddings is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=gemini_key)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30.0)
    target_collection = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")

    return QdrantVectorStore(
        client=client,
        collection_name=target_collection,
        embedding=embeddings,
    )


# ------------------------------------------------------------------------------
# 4. LLM Factory (Primary Groq + Fallback Gemini)
# ------------------------------------------------------------------------------

def build_llm_chains():
    """
    Build structured generation chains with Groq as primary and Google Gemini
    configured via .with_fallbacks().
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq

    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")
    if not gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment variables.")

    # Primary generator: Groq llama-3.3-70b-versatile
    primary_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.0,
    )

    # Fallback generator: Google Gemini
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    fallback_llm = ChatGoogleGenerativeAI(
        model=gemini_model,
        google_api_key=gemini_api_key,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert enterprise research assistant adhering to strict verification standards.\n"
            "Synthesize an answer to the user's question solely from the verified document context below.\n\n"
            "CITATION & FACTUALITY RULES:\n"
            "1. Base your answer strictly on facts explicitly contained within the `<document_context>` XML tags.\n"
            "2. If the context does not contain enough evidence to answer truthfully and comprehensively, "
            "set `is_answerable` to False, set `answer` to 'I cannot find the answer to this question in the document.', "
            "and leave `source_pages` and `source_sections` empty.\n"
            "3. When `is_answerable` is True, extract every source page number from `[Page: X]` into `source_pages` (as integers).\n"
            "4. Extract every source section header from `[Section: Y]` into `source_sections` (as strings).\n"
            "5. If the user refers to previous context ('it', 'that', 'they', 'the previous year'), use the recent conversation history to resolve the reference, while ensuring every factual claim is strictly supported by the document context.\n"
            "6. Never use outside knowledge or introduce ungrounded speculation.\n\n"
            "<recent_conversation_history>\n{history}\n</recent_conversation_history>\n\n"
            "<document_context>\n{context}\n</document_context>",
        ),
        ("human", "{question}"),
    ])

    primary_structured = primary_llm.with_structured_output(CitedAnswer)
    fallback_structured = fallback_llm.with_structured_output(CitedAnswer)

    primary_chain = prompt | primary_structured
    fallback_chain = prompt | fallback_structured

    resilient_chain = primary_chain.with_fallbacks([fallback_chain])
    return resilient_chain, prompt, primary_structured, fallback_structured


# ------------------------------------------------------------------------------
# 5. Graph Nodes & Routing Logic
# ------------------------------------------------------------------------------

def intent_classifier_node(state: GraphState) -> Dict[str, Any]:
    """
    Intent Classifier Node:
    Predicts intent to distinguish casual greetings from document inquiries.
    """
    question = state.get("question", "")
    intent = predict_intent(question)
    rag_logger.info(
        f"Intent classified: '{intent}' for query: '{question}'",
        extra={"metadata": {"intent": intent, "query": question}},
    )
    return {"intent": intent}


def route_intent(state: GraphState) -> str:
    """Conditional Edge: Route based on intent."""
    intent = state.get("intent", "document_query")
    if intent == "greeting":
        return "greeting"
    return "document_query"


def greeting_node(state: GraphState) -> Dict[str, Any]:
    """
    Greeting Node:
    Provides a helpful, conversational response without document search.
    """
    rag_logger.info("Executing Greeting Node (conversational intent).")
    ans = CitedAnswer(
        is_answerable=True,
        answer=GREETING_MESSAGE,
        source_pages=[],
        source_sections=[],
    )
    return {
        "cited_answer": ans,
        "messages": [AIMessage(content=GREETING_MESSAGE)],
    }


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve Node:
    Queries Qdrant for top-k chunks and logs cosine similarity metrics.
    """
    question = state.get("question")
    if not question and state.get("messages"):
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
                question = str(m.content)
                break
    question = question or ""

    start_time = time.perf_counter()
    documents: List[Document] = []
    similarity_scores: List[float] = []

    try:
        vector_store = get_vector_store()
        top_k = int(os.getenv("TOP_K_CHUNKS", "5"))
        results = vector_store.similarity_search_with_score(query=question, k=top_k)

        for doc, score in results:
            documents.append(doc)
            similarity_scores.append(float(score))
    except Exception as exc:
        rag_logger.error(f"Error during vector retrieval from Qdrant: {exc}", exc_info=True)

    latency_ms = (time.perf_counter() - start_time) * 1000
    log_retrieval_event(
        query=question,
        scores=similarity_scores,
        latency_ms=latency_ms,
        chunk_count=len(documents),
        top_score_threshold=SIMILARITY_THRESHOLD,
    )

    return {
        "question": question,
        "documents": documents,
        "similarity_scores": similarity_scores,
    }


def route_query(state: GraphState) -> str:
    """
    Conditional Routing Function:
    Evaluates highest similarity score against threshold (0.72).
    """
    scores = state.get("similarity_scores", [])
    max_score = max(scores) if scores else 0.0

    if max_score < SIMILARITY_THRESHOLD:
        rag_logger.info(
            f"Routing decision: REJECTION (max similarity score: {max_score:.4f} < {SIMILARITY_THRESHOLD})",
            extra={
                "similarity_scores": scores,
                "metadata": {"route": "rejection", "max_score": max_score, "threshold": SIMILARITY_THRESHOLD},
            },
        )
        return "rejection"

    rag_logger.info(
        f"Routing decision: GENERATION (max similarity score: {max_score:.4f} >= {SIMILARITY_THRESHOLD})",
        extra={
            "similarity_scores": scores,
            "metadata": {"route": "generation", "max_score": max_score, "threshold": SIMILARITY_THRESHOLD},
        },
    )
    return "generation"


def rejection_node(state: GraphState) -> Dict[str, Any]:
    """
    Rejection Node:
    Outputs standard rejection message deterministically without LLM inference.
    """
    rag_logger.info("Executing Rejection Node (query failed similarity threshold).")
    answer_obj = CitedAnswer(
        is_answerable=False,
        answer=REJECTION_MESSAGE,
        source_pages=[],
        source_sections=[],
    )
    return {
        "cited_answer": answer_obj,
        "messages": [AIMessage(content=REJECTION_MESSAGE)],
    }


def generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Generation Node:
    Prepends metadata into chunks, builds conversation history context,
    and invokes Groq with Gemini fallback.
    """
    documents = state.get("documents", [])
    question = state.get("question", "")

    # Format chunks with provenance tags
    formatted_chunks: List[str] = []
    for doc in documents:
        page = doc.metadata.get("page", "Unknown")
        section = doc.metadata.get("section", "Unknown")
        content = doc.page_content.strip()
        formatted_chunks.append(f"[Page: {page} | Section: {section}]\n{content}")

    context_str = "\n\n---\n\n".join(formatted_chunks)

    # Format multi-turn conversation history
    history_turns: List[str] = []
    if state.get("messages"):
        for msg in state["messages"][:-1]:
            sender = "User" if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human" else "Assistant"
            text = str(msg.content)[:300].strip()
            history_turns.append(f"{sender}: {text}")
    history_str = "\n".join(history_turns[-6:]) if history_turns else "No prior conversation."

    start_time = time.perf_counter()
    used_model = "llama-3.3-70b-versatile"
    is_fallback = False
    cited_answer: Optional[CitedAnswer] = None

    try:
        resilient_chain, prompt, primary_structured, fallback_structured = build_llm_chains()

        try:
            primary_chain = prompt | primary_structured
            cited_answer = primary_chain.invoke({
                "context": context_str,
                "history": history_str,
                "question": question,
            })
        except Exception as groq_err:
            rag_logger.warning(
                f"Primary Groq generation failed: {groq_err}. Triggering Gemini fallback...",
                extra={"metadata": {"primary_error": str(groq_err)}},
            )
            is_fallback = True
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            used_model = gemini_model
            fallback_chain = prompt | fallback_structured
            cited_answer = fallback_chain.invoke({
                "context": context_str,
                "history": history_str,
                "question": question,
            })
    except Exception as fatal_err:
        rag_logger.error(f"Both primary and fallback LLMs encountered an error: {fatal_err}", exc_info=True)
        cited_answer = CitedAnswer(
            is_answerable=False,
            answer=REJECTION_MESSAGE,
            source_pages=[],
            source_sections=[],
        )

    latency_ms = (time.perf_counter() - start_time) * 1000

    if not cited_answer.is_answerable:
        cited_answer.answer = REJECTION_MESSAGE
        cited_answer.source_pages = []
        cited_answer.source_sections = []

    log_generation_event(
        model_name=used_model,
        latency_ms=latency_ms,
        is_fallback=is_fallback,
        is_answerable=cited_answer.is_answerable,
        citations_count=len(cited_answer.source_pages),
    )

    return {
        "cited_answer": cited_answer,
        "messages": [AIMessage(content=cited_answer.answer)],
    }


# ------------------------------------------------------------------------------
# 6. LangGraph Assembly
# ------------------------------------------------------------------------------

def create_rag_graph() -> StateGraph:
    """Compile the LangGraph state machine with intent classification and conditional routing."""
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rejection", rejection_node)
    workflow.add_node("generation", generation_node)

    # Register edges
    workflow.add_edge(START, "intent_classifier")
    workflow.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "greeting": "greeting",
            "document_query": "retrieve",
        },
    )
    workflow.add_conditional_edges(
        "retrieve",
        route_query,
        {
            "rejection": "rejection",
            "generation": "generation",
        },
    )
    workflow.add_edge("greeting", END)
    workflow.add_edge("rejection", END)
    workflow.add_edge("generation", END)

    return workflow.compile()


# Export compiled graph application
rag_app = create_rag_graph()


if __name__ == "__main__":
    import sys

    test_query = sys.argv[1] if len(sys.argv) > 1 else "hello"
    print(f"Running LangGraph RAG evaluation on test query: '{test_query}'")
    initial_state: GraphState = {
        "messages": [HumanMessage(content=test_query)],
        "question": test_query,
        "intent": None,
        "documents": [],
        "similarity_scores": [],
        "cited_answer": None,
    }
    result = rag_app.invoke(initial_state)
    print("\nResult Answer:")
    print(result.get("cited_answer"))
