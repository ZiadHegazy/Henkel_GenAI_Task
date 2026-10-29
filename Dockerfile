# Use official Python 3.11 slim runtime as base image
FROM python:3.11-slim

# Set environment variables for optimized Python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for OCR, PDF rendering, and OpenCV headless image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pre-install lightweight CPU-only PyTorch to PREVENT downloading 4+ GB of NVIDIA CUDA / Triton GPU runtimes
RUN pip install  torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Pre-install headless OpenCV so cv2 runs in server environments without desktop OpenGL display
RUN pip install  opencv-python-headless

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install dependencies (unstructured[pdf] uses pre-installed CPU PyTorch without GPU/CUDA bloat)
RUN pip install  --prefer-binary -r requirements.txt

# Copy application source files
COPY . .

# Expose port 8000 for Chainlit web server
EXPOSE 8000

# Start Chainlit application bound to 0.0.0.0 on port 8000
CMD ["chainlit", "run", "app.py", "--port", "8000", "--host", "0.0.0.0"]
