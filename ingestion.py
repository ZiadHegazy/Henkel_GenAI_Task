"""
ingestion.py - Stateful PDF Ingestion & Vector Indexing Pipeline for Production RAG
===================================================================================

Architecture & Design Justification:
-------------------------------------
1. Strategy & Chunking Rationale (size=500, overlap=50):
   - In production RAG systems, embedding quality and retrieval precision degrade
     when chunk sizes are either too large (diluting specific facts and causing the
     "lost in the middle" phenomenon) or too small (losing syntactic coherence and
     contextual meaning).
   - A chunk size of 500 characters (~75-100 words) captures self-contained factual
     propositions, table entries, and specific policy or technical clauses without
     overwhelming the embedding model with unrelated topics.
   - A 50-character sliding overlap (~10% buffer) preserves sentence boundaries and
     semantic continuity across adjacent chunks, preventing critical keyword splits
     at arbitrary cutoffs.

2. Page-Boundary Preservation for 1-to-1 Citation Integrity:
   - Traditional text splitters ingest an entire document as a single contiguous stream,
     allowing chunks to span across page breaks. This causes attribution ambiguity,
     where a generated citation might claim Page 3 for information that actually originated
     on Page 4.
   - To guarantee strict, verifiable 1-to-1 page citations required for auditability
     and compliance, this pipeline enforces hard page-boundary flushing: chunks are
     never allowed to straddle multiple pages. Every generated vector chunk maps
     deterministically to exactly one source page (`metadata['page']`).

3. Stateful Section Tracking (Hierarchical Provenance):
   - Documents are structured hierarchically (e.g., Executive Summary, Methodologies,
     Financial Obligations, Terms of Service).
   - By parsing document elements with `unstructured` and tracking state transitions
     on title and header elements, every paragraph and table row inherits its active
     section context (`metadata['section']`).
   - When fed into the downstream LLM generation node, this section metadata is prepended
     into the prompt context, dramatically reducing hallucinations by grounding the
     model in both physical location (page) and logical structure (section).

4. High-Resolution Document Partitioning:
   - Using `unstructured.partition_pdf(strategy="hi_res")` provides layout-aware
     element classification (distinguishing Titles, NarrativeText, Tables, Headers,
     and Footers) powered by OCR/layout detection, outperforming naive PyPDF/pdfplumber
     dumps on complex multi-column or scanned layouts.

5. Embeddings & Vector Store:
   - Embeddings: Google AI Studio `text-embedding-004` (768-dimensional, high semantic
     fidelity, optimized for retrieval benchmarks).
   - Vector Index: Cloud-hosted Qdrant with Cosine Distance, providing millisecond-range
     approximate nearest neighbor (ANN) search with native payload filtering.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingestion")


def get_required_env_var(name: str) -> str:
    """Retrieve an environment variable or raise a descriptive error."""
    val = os.getenv(name)
    if not val:
        raise ValueError(
            f"Missing required environment variable: '{name}'. "
            f"Please populate it in your .env file or environment."
        )
    return val


def parse_pdf_elements(pdf_path: str) -> List[Any]:
    """
    Parse a local PDF file into structured layout elements using unstructured hi_res strategy.

    Parameters:
        pdf_path: Path to the target PDF file on disk.

    Returns:
        List of structured elements (e.g., Title, NarrativeText, ListItem, Table).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Target PDF file not found at: {pdf_path}")

    logger.info("Starting high-resolution PDF partitioning for: %s", pdf_path)
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        logger.error("Failed to import 'unstructured'. Ensure 'unstructured[pdf]' is installed.")
        raise exc

    # Attempt hi-res partitioning first; fall back gracefully to fast strategy if layout model encounters issues
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            include_page_breaks=True,
        )
    except Exception as exc:
        logger.warning(
            "Hi-res partitioning encountered an issue (%s). Falling back to strategy='fast'...",
            exc,
        )
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            include_page_breaks=True,
        )
    logger.info("Successfully partitioned PDF into %d raw structural elements.", len(elements))
    return elements


def extract_and_group_elements(
    elements: List[Any],
    default_section: str = "Introduction / Document Header",
) -> List[Dict[str, Any]]:
    """
    Iterate through unstructured elements, maintain stateful section tracking,
    and group text into buffers strictly respecting page boundaries.

    Chunking Design Guarantee:
    - Buffers NEVER span multiple pages.
    - Each buffer maintains the currently active section title.
    - Header/Title elements update the active `current_section` state.

    Parameters:
        elements: Raw elements parsed by unstructured.
        default_section: Fallback section name before the first header is encountered.

    Returns:
        A list of dictionaries with structure:
        [
            {"page": int, "section": str, "text": str},
            ...
        ]
    """
    header_categories = {
        "title",
        "header",
        "headline",
        "subheadline",
        "section-header",
    }

    current_section = default_section
    current_page: Optional[int] = None
    current_text_buffer: List[str] = []
    grouped_blocks: List[Dict[str, Any]] = []

    def flush_buffer(page: Optional[int], section: str, buffer: List[str]) -> None:
        if page is not None and buffer:
            content = "\n\n".join(segment.strip() for segment in buffer if segment.strip())
            if content:
                grouped_blocks.append({
                    "page": page,
                    "section": section,
                    "text": content,
                })

    for element in elements:
        category = str(getattr(element, "category", "") or element.__class__.__name__).lower()
        text = str(getattr(element, "text", "") or "").strip()

        # Resolve element page number
        meta = getattr(element, "metadata", None)
        elem_page = getattr(meta, "page_number", None) if meta else None
        if elem_page is None:
            elem_page = current_page if current_page is not None else 1

        if not text:
            continue

        # Check if element represents a section header or document title
        if category in header_categories:
            # Flush existing buffer before section transition
            flush_buffer(current_page, current_section, current_text_buffer)
            current_text_buffer = []

            # Update stateful section pointer
            current_section = text
            current_page = elem_page
            # Include the header text in the new section buffer for semantic context
            current_text_buffer.append(text)
        else:
            # Check if page boundary has changed
            if current_page is not None and elem_page != current_page:
                # Flush existing buffer to guarantee strict page boundary isolation
                flush_buffer(current_page, current_section, current_text_buffer)
                current_text_buffer = []

            current_page = elem_page
            current_text_buffer.append(text)

    # Flush any remaining buffer content
    flush_buffer(current_page, current_section, current_text_buffer)

    logger.info(
        "Grouped raw elements into %d page-bounded, section-tagged text blocks.",
        len(grouped_blocks),
    )
    return grouped_blocks


def split_blocks_into_chunks(
    grouped_blocks: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Any]:
    """
    Split page-bounded blocks using RecursiveCharacterTextSplitter and attach
    strict page and section metadata to every resulting chunk.

    Parameters:
        grouped_blocks: Output from `extract_and_group_elements`.
        chunk_size: Maximum chunk size in characters (default: 500).
        chunk_overlap: Sliding overlap between chunks in characters (default: 50).

    Returns:
        List of LangChain `Document` objects with populated page_content and metadata.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            keep_separator=True,
        )
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "; ", ", ", " "],
                keep_separator=True,
            )
        except ImportError:
            logger.warning(
                "Neither langchain_text_splitters nor langchain.text_splitter found. "
                "Using built-in robust character splitter."
            )
            class SimpleRecursiveSplitter:
                def __init__(self, size: int, overlap: int):
                    self.size = size
                    self.overlap = overlap
                def split_text(self, txt: str) -> List[str]:
                    res = []
                    start = 0
                    length = len(txt)
                    step = max(1, self.size - self.overlap)
                    while start < length:
                        end = min(start + self.size, length)
                        res.append(txt[start:end])
                        if end == length:
                            break
                        start += step
                    return res
            splitter = SimpleRecursiveSplitter(size=chunk_size, overlap=chunk_overlap)

    from langchain_core.documents import Document

    all_chunks: List[Document] = []

    for block in grouped_blocks:
        page_num = block["page"]
        section_name = block["section"]
        raw_text = block["text"]

        splits = splitter.split_text(raw_text)
        for chunk_idx, chunk_text in enumerate(splits):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "page": int(page_num),
                    "section": str(section_name),
                    "chunk_id": chunk_idx,
                },
            )
            all_chunks.append(chunk_doc)

    logger.info(
        "Split %d text blocks into %d discrete chunks (chunk_size=%d, overlap=%d).",
        len(grouped_blocks),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks


def load_chunks_into_qdrant(
    documents: List[Any],
    collection_name: str = "rag_documents",
    recreate_collection: bool = False,
    batch_size: int = 64,
) -> Any:
    """
    Embed chunks using GoogleGenerativeAIEmbeddings (text-embedding-004) and index
    into Qdrant Cloud.

    Parameters:
        documents: List of LangChain Document objects.
        collection_name: Name of target Qdrant collection.
        recreate_collection: If True, drops and recreates the collection.
        batch_size: Batch size for embedding and indexing operations.

    Returns:
        An instantiated QdrantVectorStore connected to the populated collection.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models

    qdrant_url = get_required_env_var("QDRANT_URL")
    qdrant_api_key = get_required_env_var("QDRANT_API_KEY")

    # Support either GEMINI_API_KEY or GOOGLE_API_KEY
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY for embedding generation.")
    os.environ["GOOGLE_API_KEY"] = gemini_key

    logger.info("Initializing GoogleGenerativeAIEmbeddings (models/text-embedding-004)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=gemini_key,
    )

    logger.info("Connecting to Qdrant cluster at: %s", qdrant_url)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60.0)

    # Check collection existence
    existing_collections = [col.name for col in client.get_collections().collections]
    logger.info("Existing Qdrant collections: %s", existing_collections)

    if collection_name in existing_collections and recreate_collection:
        logger.warning("Recreating collection '%s' (dropping existing data)...", collection_name)
        client.delete_collection(collection_name=collection_name)
        existing_collections.remove(collection_name)

    if collection_name not in existing_collections:
        logger.info(
            "Creating Qdrant collection '%s' with 768-dim Cosine distance vectors...",
            collection_name,
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest_models.VectorParams(
                size=768,  # text-embedding-004 dimension
                distance=rest_models.Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    total_docs = len(documents)
    logger.info("Uploading %d chunks to Qdrant collection '%s'...", total_docs, collection_name)

    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(documents=batch)
        logger.info("Indexed chunks %d to %d of %d.", i + 1, min(i + batch_size, total_docs), total_docs)

    logger.info("Successfully indexed all %d chunks into Qdrant collection '%s'.", total_docs, collection_name)
    return vector_store


def run_pipeline(
    pdf_path: str,
    collection_name: str = "rag_documents",
    recreate_collection: bool = False,
) -> None:
    """Execute the end-to-end PDF ingestion pipeline."""
    logger.info("=== Starting PDF Ingestion Pipeline ===")
    logger.info("File: %s", pdf_path)
    logger.info("Target Collection: %s", collection_name)

    # 1. Parse PDF with hi_res strategy
    elements = parse_pdf_elements(pdf_path)

    # 2. Stateful section tracking and page-boundary buffer grouping
    grouped_blocks = extract_and_group_elements(elements)

    # 3. Recursive chunking (size=500, overlap=50) with metadata attachment
    chunks = split_blocks_into_chunks(grouped_blocks, chunk_size=500, chunk_overlap=50)

    if not chunks:
        logger.warning("No chunks were generated from the document. Exiting.")
        return

    # 4. Embed with text-embedding-004 and index into Qdrant
    load_chunks_into_qdrant(
        documents=chunks,
        collection_name=collection_name,
        recreate_collection=recreate_collection,
    )
    logger.info("=== PDF Ingestion Pipeline Completed Successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stateful PDF Ingestion Pipeline with Metadata Tracking for Production RAG"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to the PDF file to ingest",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=os.getenv("QDRANT_COLLECTION_NAME", "rag_documents"),
        help="Target Qdrant collection name (default: rag_documents or QDRANT_COLLECTION_NAME env)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection if it already exists",
    )

    args = parser.parse_args()
    run_pipeline(
        pdf_path=args.file,
        collection_name=args.collection,
        recreate_collection=args.recreate,
    )
