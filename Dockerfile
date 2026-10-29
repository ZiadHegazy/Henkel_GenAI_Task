# Use official Python 3.11 slim runtime as base image
FROM python:3.11-slim

# Set environment variables for optimized Python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory inside container
WORKDIR /app

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install lightweight dependencies utilizing pip caching and binary wheels
RUN pip install --prefer-binary --timeout 120 --retries 5 -r requirements.txt

# Copy application source files
COPY . .

# Expose port 8000 for Chainlit web server
EXPOSE 8000

# Start Chainlit application bound to 0.0.0.0 on port 8000
CMD ["chainlit", "run", "app.py", "--port", "8000", "--host", "0.0.0.0"]
