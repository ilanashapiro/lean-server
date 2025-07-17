FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl git unzip build-essential libgmp-dev \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy JUST the kimina server so the image can be cached during rebuilds
COPY kimina-lean-server ./kimina-lean-server

# Set environment variable for the wrapper
ENV KIMINA_HOST=http://localhost:12332

# Go into actual Kimina server directory
WORKDIR /app/kimina-lean-server

# Install Lean via elan
RUN curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y

# Add Lean tools to PATH
ENV PATH="/root/.elan/bin:$PATH"

# Copy env template
RUN cp .env.template .env

# Install dependencies
RUN pip install -e .

# Setup Lean env
RUN bash setup.sh

# copy the rest of the Lean server wrapper code
# Set working directory back to /app/lean-server before copying the wrapper
WORKDIR /app
COPY wrapper_server ./wrapper_server


# ~6min for first run