FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl git unzip build-essential libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy whole wrapper folder into container
COPY LeanDockerWrapper ./lean-server

# Set environment variable for the wrapper
ENV KIMINA_HOST=http://host.docker.internal:12332

# Go into actual Kimina server directory
WORKDIR /app/lean-server/kimina-lean-server

# Copy env template
RUN cp .env.template .env

# Install dependencies
RUN pip install -e .

# Setup Lean env
RUN bash setup.sh

# Expose default FastAPI port (if needed)
EXPOSE 80

# took ~6min to build