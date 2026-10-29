# Use official Python 3.11 slim runtime as base image
FROM python:3.11-slim

# Set environment variables for optimized Python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for OCR and document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for layer caching
COPY requirements.txt .

# Exclude offline ingestion package (unstructured[pdf]) during container build
# to prevent downloading 4+ GB of heavy local GPU runtimes (nvidia-cudnn, cuda-toolkit, triton)
# since the runtime application uses cloud APIs (Groq, Google AI Studio, Qdrant Cloud).
RUN sed '/unstructured/d' requirements.txt > /tmp/runtime-requirements.txt \
    && pip install --no-cache-dir --prefer-binary -r /tmp/runtime-requirements.txt \
    && rm /tmp/runtime-requirements.txt

# Copy application source files
COPY . .

# Expose port 8000 for Chainlit web server
EXPOSE 8000

# Start Chainlit application bound to 0.0.0.0 on port 8000
CMD ["chainlit", "run", "app.py", "--port", "8000", "--host", "0.0.0.0"]
