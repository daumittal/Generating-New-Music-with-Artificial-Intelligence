# Builder stage: Install dependencies and prepare the application
FROM python:3.9-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Preprocess (e.g., verify or bundle assets)
RUN mkdir -p /app/models && \
    python -m scripts.preprocess --output-dir /app/models

# Runner stage: Create lightweight runtime
FROM python:3.9-slim AS runner
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts from builder
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.9 /usr/local/lib/python3.9

# Set working directory and environment
WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Entry point to run the music generation script
ENTRYPOINT ["/bin/sh", "-c", "python -m scripts.generate_music \"$@\"", "--"]