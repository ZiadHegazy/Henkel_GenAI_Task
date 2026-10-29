"""
logger.py - Production JSON Logger for RAG Observability
========================================================

Provides structured, machine-readable JSON logging to stdout, tracking:
- Query latency (retrieval and generation milliseconds)
- Vector similarity scores (cosine distance metrics)
- Token consumption (prompt, completion, total)
- Structured execution metadata and exception traces
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON strings formatted for enterprise log
    aggregators (Datadog, CloudWatch, Elasticsearch, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Track query / node latency
        if hasattr(record, "latency_ms"):
            log_record["latency_ms"] = round(float(getattr(record, "latency_ms")), 2)
        elif hasattr(record, "latency"):
            log_record["latency_ms"] = round(float(getattr(record, "latency")) * 1000, 2)

        # Track vector similarity scores
        if hasattr(record, "similarity_scores"):
            scores = getattr(record, "similarity_scores")
            if isinstance(scores, list):
                log_record["similarity_scores"] = [round(float(s), 4) for s in scores]
                if scores:
                    log_record["max_similarity_score"] = round(float(max(scores)), 4)
                    log_record["min_similarity_score"] = round(float(min(scores)), 4)

        # Track LLM token consumption
        if hasattr(record, "tokens") and isinstance(record.tokens, dict):
            log_record["tokens"] = record.tokens

        # Track additional custom metadata
        if hasattr(record, "metadata") and isinstance(record.metadata, dict):
            log_record["metadata"] = record.metadata

        # Format exceptions if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)


def get_json_logger(name: str = "rag_pipeline", level: int = logging.INFO) -> logging.Logger:
    """
    Initialize and return a configured JSON logger sending formatted output to stdout.
    Avoids duplicate handlers if invoked multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if a StreamHandler with JSONFormatter is already attached
    already_configured = any(
        isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JSONFormatter)
        for h in logger.handlers
    )

    if not already_configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# Global application logger
rag_logger = get_json_logger("rag_orchestration")


def log_retrieval_event(
    query: str,
    scores: List[float],
    latency_ms: float,
    chunk_count: int,
    top_score_threshold: float = 0.72,
) -> None:
    """Helper to log document retrieval results with similarity scores and latency."""
    max_score = max(scores) if scores else 0.0
    passed_threshold = max_score >= top_score_threshold

    rag_logger.info(
        f"Retrieved {chunk_count} candidate chunks (max_score: {max_score:.4f}, threshold: {top_score_threshold})",
        extra={
            "latency_ms": latency_ms,
            "similarity_scores": scores,
            "metadata": {
                "query": query,
                "chunk_count": chunk_count,
                "max_score": round(max_score, 4),
                "threshold": top_score_threshold,
                "passed_threshold": passed_threshold,
            },
        },
    )


def log_generation_event(
    model_name: str,
    latency_ms: float,
    tokens: Optional[Dict[str, int]] = None,
    is_fallback: bool = False,
    is_answerable: bool = True,
    citations_count: int = 0,
) -> None:
    """Helper to log generation results with latency, token usage, and provenance count."""
    rag_logger.info(
        f"Generation finished using {model_name}" + (" [FALLBACK]" if is_fallback else ""),
        extra={
            "latency_ms": latency_ms,
            "tokens": tokens or {},
            "metadata": {
                "model": model_name,
                "is_fallback": is_fallback,
                "is_answerable": is_answerable,
                "citations_count": citations_count,
            },
        },
    )
