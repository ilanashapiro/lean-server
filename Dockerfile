FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl git unzip build-essential libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir to top-level
WORKDIR /app

# Copy kimina repo (assuming it’s in the build context)
COPY kimina-lean-server ./kimina-lean-server

# Setup env
WORKDIR /app/kimina-lean-server
RUN cp .env.template .env

# Install Python dependencies
RUN pip install -e .

# Run Lean setup
RUN bash setup.sh

# Expose FastAPI port
EXPOSE 80

# took ~6min to build