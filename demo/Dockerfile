# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    portaudio19-dev \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the demo application code
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Expose the ports that the applications use
EXPOSE 8080 8090

# Use the start script to run both processes with prefixed logging
CMD ["./start.sh"]
