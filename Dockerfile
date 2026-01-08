FROM python:3.11.4-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code including .env
COPY . .

# Set environment variables explicitly in Docker
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/app/models

# Create necessary directories with proper permissions
RUN mkdir -p models temp chroma_db && \
    chmod 755 models temp chroma_db

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]