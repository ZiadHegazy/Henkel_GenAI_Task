"""
ingestion.py - Fast Digital PDF Ingestion & Vector Indexing Pipeline
=====================================================================

Architecture & Design Justification (Text-Only / Fast Mode):
------------------------------------------------------------
1. Zero-OCR & Digital Text Rationale:
   - This pipeline assumes input documents contain digital text streams.
   - All image elements, figures, and graphical decorations are explicitly bypassed,
     eliminating the need for heavyweight OCR engines (Tesseract, Poppler), computer vision
     frameworks (OpenCV, PyTorch, detectron2), and Hugging Face model downloads.
   - Uses `unstructured.partition_pdf(strategy="fast")` backed by direct PDF text stream
     extraction, delivering sub-second parsing with zero GPU/CUDA overhead.
   - Includes a native `pypdf` fallback to ensure deterministic execution under any environment.

2. Page-Boundary Preservation for 1-to-1 Citation Integrity:
   - To guarantee strict, verifiable 1-to-1 page citations required for auditability,
     buffers are flushed whenever `page_number` transitions.
   - Chunks never straddle multiple pages, ensuring every generated chunk maps
     deterministically to exactly one source page (`metadata['page']`).

3. Stateful Section Tracking (Hierarchical Provenance):
   - Maintains an active section pointer (`current_section`) updated whenever a Title,
     Header, or Section Heading is encountered.
   - All text paragraphs inherit their active section context (`metadata['section']`).

4. Embeddings & Vector Index:
   - Google AI Studio `text-embedding-004` (768-dimensional normalized vectors).
   - Qdrant Cloud vector database with Cosine Distance.
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
    Parse a local PDF file into structured elements using fast digital text extraction.
    Assumes PDF contains digital text only; all image elements are explicitly bypassed.
    Does NOT require OCR, Tesseract, Poppler, or GPU models.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Target PDF file not found at: {pdf_path}")

    logger.info("Starting fast text extraction (no OCR, bypassing images) for: %s", pdf_path)

    # 1. Primary parser: unstructured with strategy='fast'
    try:
        from unstructured.partition.pdf import partition_pdf

        raw_elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            include_page_breaks=True,
        )

        # Filter out images, figures, and empty elements
        filtered_elements = []
        for el in raw_elements:
            cat = str(getattr(el, "category", "") or el.__class__.__name__).lower()
            if cat in ("image", "figure", "picture", "drawing"):
                continue  # Bypassed per specification
            filtered_elements.append(el)

        logger.info(
            "Fast unstructured parsing extracted %d text elements (bypassed image elements).",
            len(filtered_elements),
        )
        return filtered_elements

    except Exception as exc:
        logger.warning(
            "Unstructured fast partition encountered an issue (%s). Using native pypdf fallback...",
            exc,
        )

    # 2. Resilient Fallback: native pypdf page extraction
    try:
        from pypdf import PdfReader
    except ImportError:
        import pypdf
        PdfReader = pypdf.PdfReader

    reader = PdfReader(pdf_path)
    fallback_elements = []

    class MockElement:
        def __init__(self, text: str, page_number: int, category: str):
            self.text = text
            self.category = category
            self.metadata = type("Meta", (), {"page_number": page_number})()

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]

        for p_idx, para in enumerate(paragraphs):
            # Simple heuristic: short lines at top or capitalized lines are headers
            lines = para.split("\n")
            first_line = lines[0].strip()
            if (len(first_line) < 80 and (first_line.isupper() or first_line.istitle())) or p_idx == 0:
                fallback_elements.append(MockElement(text=first_line, page_number=page_idx, category="Title"))
                remaining_text = "\n".join(lines[1:]).strip()
                if remaining_text:
                    fallback_elements.append(
                        MockElement(text=remaining_text, page_number=page_idx, category="NarrativeText")
                    )
            else:
                fallback_elements.append(MockElement(text=para, page_number=page_idx, category="NarrativeText"))

    logger.info("Native pypdf extraction completed with %d text elements.", len(fallback_elements))
    return fallback_elements


def extract_and_group_elements(
    elements: List[Any],
    default_section: str = "Introduction / Document Header",
) -> List[Dict[str, Any]]:
    """
    Iterate through elements, maintain stateful section tracking,
    and group text into buffers strictly respecting page boundaries.

    Guarantees:
    - Images are ignored.
    - Buffers NEVER span multiple pages.
    - Each buffer maintains the currently active section title.
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

        # Explicitly bypass any image or graphical elements
        if category in ("image", "figure", "picture", "drawing") or not text:
            continue

        # Resolve element page number
        meta = getattr(element, "metadata", None)
        elem_page = getattr(meta, "page_number", None) if meta else None
        if elem_page is None:
            elem_page = current_page if current_page is not None else 1

        # Check if element represents a section header or document title
        if category in header_categories:
            flush_buffer(current_page, current_section, current_text_buffer)
            current_text_buffer = []

            # Update stateful section pointer
            current_section = text
            current_page = elem_page
            current_text_buffer.append(text)
        else:
            # Check if page boundary has changed
            if current_page is not None and elem_page != current_page:
                flush_buffer(current_page, current_section, current_text_buffer)
                current_text_buffer = []

            current_page = elem_page
            current_text_buffer.append(text)

    # Flush any remaining buffer content
    flush_buffer(current_page, current_section, current_text_buffer)

    logger.info(
        "Grouped text into %d page-bounded, section-tagged blocks (1-to-1 page mapping).",
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
                "langchain text splitters not available. Using built-in character splitter."
            )
            class BuiltinSplitter:
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
            splitter = BuiltinSplitter(size=chunk_size, overlap=chunk_overlap)

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
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models

    qdrant_url = get_required_env_var("QDRANT_URL")
    qdrant_api_key = get_required_env_var("QDRANT_API_KEY")

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
        logger.info(
            "Indexed chunks %d to %d of %d.",
            i + 1,
            min(i + batch_size, total_docs),
            total_docs,
        )

    logger.info("Successfully indexed all %d chunks into Qdrant collection '%s'.", total_docs, collection_name)
    return vector_store


def run_pipeline(
    pdf_path: str,
    collection_name: str = "rag_documents",
    recreate_collection: bool = False,
) -> None:
    """Execute the end-to-end PDF ingestion pipeline."""
    logger.info("=== Starting PDF Ingestion Pipeline (Text-Only Mode) ===")
    logger.info("File: %s", pdf_path)
    logger.info("Target Collection: %s", collection_name)

    # 1. Parse PDF with fast text-only extraction (bypassing images)
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
        description="Fast Text-Only PDF Ingestion Pipeline for Production RAG (No OCR, Images Bypassed)"
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
