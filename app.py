"""
app.py - Production Chainlit UI with SQLite Multi-Session Chat Persistence,
         Intent Prediction, and Native Step Tracing
=============================================================================

Key Capabilities:
1. SQLite Chat Persistence & Multi-Session History:
   - Uses `database.py` (`SQLiteDataLayer`) to store all chat sessions in SQLite.
   - Enables left-sidebar session history, thread switching, and session continuation.
   - Resumes prior threads via `@cl.on_chat_resume`, rehydrating multi-turn memory.

2. Intent Prediction:
   - Distinguishes conversational queries ("hello", "who are you") from factual document queries.
   - Bypasses false rejections and responds warmly with assistant guidance.

3. Native LangGraph Step Tracing:
   - `cl.Step` interactive traces: "Intent Prediction", "Retrieving documents",
     "Evaluating similarity", and "Synthesizing answer".

4. Verifiable Citations & Streaming:
   - Streams answers token-by-token with explicit 1-to-1 page and section provenance.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Load environment configuration
load_dotenv()

# Import LangGraph components, state, and intent predictor
from graph import (
    CitedAnswer,
    GREETING_MESSAGE,
    GraphState,
    REJECTION_MESSAGE,
    SIMILARITY_THRESHOLD,
    generation_node,
    predict_intent,
    rag_app,
    rejection_node,
    retrieve_node,
    route_query,
)
from logger import get_json_logger
from database import sqlite_data_layer

app_logger = get_json_logger("chainlit_app")


# ------------------------------------------------------------------------------
# 1. Chainlit SQLite Data Layer (Public Access - No Login Required)
# ------------------------------------------------------------------------------

@cl.data_layer
def get_data_layer():
    """Register SQLite as the persistent storage backend for Chainlit."""
    return sqlite_data_layer


# ------------------------------------------------------------------------------
# 2. Session Lifecycle & Resumption Handlers
# ------------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """
    Session Initialization for a brand new conversation.
    Sets up SQLite tables and initializes in-memory state.
    """
    await sqlite_data_layer.setup()
    app_logger.info("New Chainlit conversation started.")

    cl.user_session.set("graph", rag_app)
    cl.user_session.set("history", [])

    welcome_text = (
        "## 📚 Enterprise Verifiable RAG Assistant\n\n"
        "Welcome! I am ready to answer questions strictly grounded in your indexed documents.\n\n"
        "**System Capabilities:**\n"
        "- **🧠 Intent Prediction:** Recognizes conversational greetings vs. document inquiries\n"
        "- **💾 SQLite Persistence:** All chat sessions are saved and can be resumed from the left sidebar\n"
        "- **🛡️ Quality Gate:** Cosine threshold $\\ge 0.72$ prevents hallucinations\n"
        "- **⚡ Dual LLMs:** Primary Groq `llama-3.3-70b` with Google `gemini-2.5-flash` fallback\n"
        "- **📍 Provenance:** Strict 1-to-1 page citations and section headers\n\n"
        "*Type your question below or start with 'Hello' to begin.*"
    )

    await cl.Message(content=welcome_text).send()


@cl.on_chat_resume
async def on_chat_resume(thread: Dict[str, Any]):
    """
    Triggered when a user clicks an existing chat session in the sidebar.
    Restores the thread's conversation history into memory for seamless multi-turn continuity.
    """
    thread_id = thread.get("id")
    app_logger.info(f"Resuming existing chat session: {thread_id}")

    # Reconstruct multi-turn memory from persisted steps
    restored_history: List[BaseMessage] = []
    steps = thread.get("steps", [])

    for s in steps:
        step_type = s.get("type")
        output = s.get("output", "")
        if step_type == "user_message" and output:
            restored_history.append(HumanMessage(content=output))
        elif step_type == "assistant_message" and output:
            # Strip citation footer from LLM memory context
            clean_output = output.split("\n\n---")[0].strip()
            restored_history.append(AIMessage(content=clean_output))

    cl.user_session.set("history", restored_history)
    cl.user_session.set("graph", rag_app)

    await cl.Message(
        content=f"*Session successfully resumed from database ({len(restored_history)} prior turns loaded into memory).*"
    ).send()


# ------------------------------------------------------------------------------
# 3. Message Processing with Intent Prediction & Multi-Turn State
# ------------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    """
    Message Handler:
    1. Predicts intent (greeting vs. document query).
    2. If greeting: responds warmly without false document rejection.
    3. If document query: executes LangGraph with Qdrant retrieval, quality gating,
       and Groq/Gemini generation with multi-turn conversation memory.
    """
    user_query = message.content.strip()
    if not user_query:
        return

    # Retrieve prior conversation turns for multi-turn context
    history: List[BaseMessage] = cl.user_session.get("history", [])

    # --------------------------------------------------------------------------
    # Step 1: Intent Prediction
    # --------------------------------------------------------------------------
    async with cl.Step(name="Intent Prediction", type="routing") as step:
        step.input = f"Analyzing query: '{user_query}'"
        intent = predict_intent(user_query)

        if intent == "greeting":
            step.output = (
                "👋 **Intent Detected: Greeting / Conversational**\n"
                "Routing directly to Conversational Assistant (bypassing document retrieval)."
            )
        else:
            step.output = (
                "🔎 **Intent Detected: Document Query**\n"
                "Routing to Qdrant vector retrieval with cosine quality gating."
            )

    # --------------------------------------------------------------------------
    # Branch A: Handle Conversational / Greeting Intent
    # --------------------------------------------------------------------------
    if intent == "greeting":
        greeting_msg = cl.Message(content="")
        await greeting_msg.send()

        # Stream greeting response smoothly
        words = GREETING_MESSAGE.split(" ")
        for idx, word in enumerate(words):
            await greeting_msg.stream_token(word + (" " if idx < len(words) - 1 else ""))
            await asyncio.sleep(0.01)
        await greeting_msg.update()

        # Persist conversation turn in memory
        history.append(HumanMessage(content=user_query))
        history.append(AIMessage(content=GREETING_MESSAGE))
        cl.user_session.set("history", history)
        return

    # --------------------------------------------------------------------------
    # Branch B: Handle Document Query (RAG State Machine)
    # --------------------------------------------------------------------------
    current_messages = history + [HumanMessage(content=user_query)]

    state: GraphState = {
        "messages": current_messages,
        "question": user_query,
        "intent": intent,
        "documents": [],
        "similarity_scores": [],
        "cited_answer": None,
    }

    # Step 2: Retrieving Documents from Qdrant
    async with cl.Step(name="Retrieving documents", type="retrieval") as step:
        step.input = f"Query: {user_query}"

        retrieval_output = await asyncio.to_thread(retrieve_node, state)
        state["documents"] = retrieval_output.get("documents", [])
        state["similarity_scores"] = retrieval_output.get("similarity_scores", [])

        docs_count = len(state["documents"])
        scores = state["similarity_scores"]
        max_score = max(scores) if scores else 0.0

        score_details = ", ".join(f"{s:.4f}" for s in scores) if scores else "None"
        step.output = (
            f"Retrieved {docs_count} candidate chunks from Qdrant Cloud.\n"
            f"- Cosine Similarity Scores: [{score_details}]\n"
            f"- Highest Score: {max_score:.4f}"
        )

    # Step 3: Evaluating Similarity (Quality Gate)
    async with cl.Step(name="Evaluating similarity", type="routing") as step:
        step.input = f"Top Score: {max_score:.4f} vs Threshold: {SIMILARITY_THRESHOLD}"

        route = route_query(state)
        if route == "rejection":
            step.output = (
                f"❌ Max similarity ({max_score:.4f}) is below the {SIMILARITY_THRESHOLD} quality threshold.\n"
                f"Routing to Rejection Node (no LLM generation)."
            )
        else:
            step.output = (
                f"✅ Max similarity ({max_score:.4f}) meets the {SIMILARITY_THRESHOLD} threshold.\n"
                f"Routing to Structured Generation Node with multi-turn memory."
            )

    # Step 4: Execution (Rejection or Generation)
    cited_answer: CitedAnswer

    if route == "rejection":
        async with cl.Step(name="Rejection", type="system") as step:
            rejection_output = await asyncio.to_thread(rejection_node, state)
            cited_answer = rejection_output["cited_answer"]
            step.output = "Deterministic rejection generated."

        await cl.Message(content=cited_answer.answer).send()

        # Update memory
        history.append(HumanMessage(content=user_query))
        history.append(AIMessage(content=cited_answer.answer))
        cl.user_session.set("history", history)
        return

    # Generation Path with LLM Fallbacks & Multi-Turn Context
    async with cl.Step(name="Synthesizing answer with citations", type="llm") as step:
        step.input = f"Context chunks: {len(state['documents'])} | Prior turns: {len(history)} | Primary: Groq Llama 3.3"
        generation_output = await asyncio.to_thread(generation_node, state)
        cited_answer = generation_output["cited_answer"]

        step.output = (
            f"Generation completed.\n"
            f"- Answerable: {cited_answer.is_answerable}\n"
            f"- Cited Pages: {cited_answer.source_pages}\n"
            f"- Cited Sections: {cited_answer.source_sections}"
        )

    # --------------------------------------------------------------------------
    # Step 5: Stream Answer & Render Citations
    # --------------------------------------------------------------------------
    if not cited_answer.is_answerable or cited_answer.answer == REJECTION_MESSAGE:
        await cl.Message(content=REJECTION_MESSAGE).send()
        history.append(HumanMessage(content=user_query))
        history.append(AIMessage(content=REJECTION_MESSAGE))
        cl.user_session.set("history", history)
        return

    # Stream the synthesized answer text
    response_msg = cl.Message(content="")
    await response_msg.send()

    words = cited_answer.answer.split(" ")
    for idx, word in enumerate(words):
        await response_msg.stream_token(word + (" " if idx < len(words) - 1 else ""))
        await asyncio.sleep(0.015)

    # Format Citation Metadata
    pages_display = (
        ", ".join(f"Page {p}" for p in cited_answer.source_pages)
        if cited_answer.source_pages
        else "No specific page cited"
    )
    sections_display = (
        ", ".join(f"'{s}'" for s in cited_answer.source_sections)
        if cited_answer.source_sections
        else "No specific section cited"
    )

    citation_footer = (
        f"\n\n---\n"
        f"📍 **Source Pages:** {pages_display}\n\n"
        f"📑 **Source Sections:** {sections_display}"
    )
    await response_msg.stream_token(citation_footer)
    await response_msg.update()

    # Render dedicated Chainlit inline Text Element for persistent inspection
    citation_element_content = (
        f"### 🔍 Verifiable Provenance Metadata\n\n"
        f"- **Source Pages (1-to-1):** {pages_display}\n"
        f"- **Source Sections:** {sections_display}\n"
        f"- **Retrieval Max Cosine Score:** {max_score:.4f}\n"
        f"- **Chunks In Context:** {len(state['documents'])}\n"
        f"- **Multi-Turn Context Turns:** {len(history)}\n"
    )

    provenance_element = cl.Text(
        name="Source Citations & Provenance",
        content=citation_element_content,
        display="inline",
    )

    await cl.Message(
        content="*(Verified with 1-to-1 page citations and section tracking)*",
        elements=[provenance_element],
    ).send()

    # Update conversation history in memory
    history.append(HumanMessage(content=user_query))
    history.append(AIMessage(content=cited_answer.answer))
    cl.user_session.set("history", history)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
