# MLOps Sentiment Analysis - Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Git LFS and build tools
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    gcc \
    g++ \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models mlruns

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PREFECT_API_URL=""

# Expose ports
EXPOSE 5000 4200

# Default command: run pipeline
CMD ["python", "pipeline/prefect_flow.py"]
