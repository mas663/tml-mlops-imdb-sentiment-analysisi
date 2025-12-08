# Docker image for MLOps Sentiment Analysis
FROM python:3.9-slim

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    gcc \
    g++ \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first for better caching
COPY requirements.txt .

# install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# create necessary directories
RUN mkdir -p data/raw data/processed models mlruns

# environment variables
ENV PYTHONUNBUFFERED=1

# expose ports
EXPOSE 5000 4200

# default command
CMD ["python", "pipeline/prefect_flow.py"]
