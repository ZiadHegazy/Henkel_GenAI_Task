"""
app.py - Production Chainlit UI with Native Step Tracing & Strict Citations
============================================================================

Features:
- Initializes LangGraph RAG application on session start (@cl.on_chat_start).
- Interactively traces execution flow using `cl.Step` wrappers:
  * "Retrieving documents" (fetches chunks from Qdrant, displays similarity scores)
  * "Evaluating similarity" (evaluates threshold 0.72 routing gate)
  * "Synthesizing answer" (executes Groq / Gemini structured generation)
- Streams response text and renders `source_pages` and `source_sections`
  using Chainlit inline `Text` elements and formatted markdown at the bottom of the response.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Load environment configuration
load_dotenv()

# Import LangGraph components and schemas
from graph import (
    CitedAnswer,
    GraphState,
    REJECTION_MESSAGE,
    SIMILARITY_THRESHOLD,
    generation_node,
    rag_app,
    rejection_node,
    retrieve_node,
    route_query,
)
from logger import get_json_logger

app_logger = get_json_logger("chainlit_app")


@cl.on_chat_start
async def on_chat_start():
    """
    Session Initialization:
    Instantiates session state and displays greeting with system configuration.
    """
    app_logger.info("New Chainlit session initiated.")
    cl.user_session.set("graph", rag_app)
    cl.user_session.set("history", [])

    welcome_text = (
        "## 📚 Enterprise Document RAG Assistant\n\n"
        "Welcome! I am ready to answer questions strictly grounded in your indexed documents.\n\n"
        "**System Architecture Highlights:**\n"
        "- **Vector Index:** Qdrant Cloud (Google `text-embedding-004`)\n"
        "- **Quality Gate:** Cosine similarity threshold $\\ge 0.72$\n"
        "- **Primary LLM:** Groq `llama-3.3-70b-versatile`\n"
        "- **Fallback LLM:** Google `gemini-2.5-flash`\n"
        "- **Citation Enforcement:** 1-to-1 page numbers & section titles\n\n"
        "*Type your question below to begin.*"
    )

    await cl.Message(content=welcome_text).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Message Handler:
    Orchestrates the LangGraph execution wrapped inside native Chainlit Steps,
    streams the synthesized answer, and renders explicit citations.
    """
    user_query = message.content.strip()
    if not user_query:
        return

    app_logger.info(f"Received user query: '{user_query}'")

    state: GraphState = {
        "messages": [HumanMessage(content=user_query)],
        "question": user_query,
        "documents": [],
        "similarity_scores": [],
        "cited_answer": None,
    }

    # --------------------------------------------------------------------------
    # Step 1: Retrieving Documents
    # --------------------------------------------------------------------------
    async with cl.Step(name="Retrieving documents", type="retrieval") as step:
        step.input = f"Query: {user_query}"

        # Execute retrieval node in background thread
        retrieval_output = await asyncio.to_thread(retrieve_node, state)
        state["documents"] = retrieval_output.get("documents", [])
        state["similarity_scores"] = retrieval_output.get("similarity_scores", [])

        docs_count = len(state["documents"])
        scores = state["similarity_scores"]
        max_score = max(scores) if scores else 0.0

        score_details = ", ".join(f"{s:.4f}" for s in scores) if scores else "None"
        step.output = (
            f"Retrieved {docs_count} candidate chunks from Qdrant.\n"
            f"- Cosine Similarity Scores: [{score_details}]\n"
            f"- Highest Score: {max_score:.4f}"
        )

    # --------------------------------------------------------------------------
    # Step 2: Evaluating Similarity (Routing Gate)
    # --------------------------------------------------------------------------
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
                f"Routing to Structured Generation Node."
            )

    # --------------------------------------------------------------------------
    # Step 3: Execution (Rejection or Generation)
    # --------------------------------------------------------------------------
    cited_answer: CitedAnswer

    if route == "rejection":
        async with cl.Step(name="Rejection", type="system") as step:
            rejection_output = await asyncio.to_thread(rejection_node, state)
            cited_answer = rejection_output["cited_answer"]
            step.output = "Rejection response generated."

        # Display rejection deterministically
        await cl.Message(content=cited_answer.answer).send()
        return

    # Generation Path with LLM Fallbacks
    async with cl.Step(name="Synthesizing answer with citations", type="llm") as step:
        step.input = f"Context chunks: {len(state['documents'])} | Primary: Groq llama-3.3 | Fallback: Gemini"
        generation_output = await asyncio.to_thread(generation_node, state)
        cited_answer = generation_output["cited_answer"]

        step.output = (
            f"Generation completed.\n"
            f"- Answerable: {cited_answer.is_answerable}\n"
            f"- Cited Pages: {cited_answer.source_pages}\n"
            f"- Cited Sections: {cited_answer.source_sections}"
        )

    # --------------------------------------------------------------------------
    # Step 4: Stream Answer & Render Citations
    # --------------------------------------------------------------------------
    if not cited_answer.is_answerable or cited_answer.answer == REJECTION_MESSAGE:
        await cl.Message(content=REJECTION_MESSAGE).send()
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

    # Append citation footer directly to the streamed message
    citation_footer = (
        f"\n\n---\n"
        f"📍 **Source Pages:** {pages_display}\n\n"
        f"📑 **Source Sections:** {sections_display}"
    )
    await response_msg.stream_token(citation_footer)
    await response_msg.update()

    # Render dedicated Chainlit inline Text Element for persistent provenance inspection
    citation_element_content = (
        f"### 🔍 Verifiable Provenance Metadata\n\n"
        f"- **Source Pages (1-to-1):** {pages_display}\n"
        f"- **Source Sections:** {sections_display}\n"
        f"- **Retrieval Max Cosine Score:** {max_score:.4f}\n"
        f"- **Chunks In Context:** {len(state['documents'])}\n"
    )

    provenance_element = cl.Text(
        name="Source Citations & Provenance",
        content=citation_element_content,
        display="inline",
    )

    # Update or send element alongside response
    await cl.Message(
        content="*(Verified with 1-to-1 page citations and section tracking)*",
        elements=[provenance_element],
    ).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
