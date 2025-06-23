# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install system dependencies including tini
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    portaudio19-dev \
    python3-dev \
    pkg-config \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install PyTorch CPU-only first (smaller and faster for containers)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Download the turn detector model files
# Set memory-efficient environment variables for model loading
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
RUN python unity/service/call.py download-files

# Set runtime environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Expose the ports that the applications use
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Use Uvicorn for production ASGI server
CMD ["uvicorn", "wrapper_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
