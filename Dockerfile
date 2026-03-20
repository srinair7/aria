FROM python:3.12-slim

# Install system dependencies for voice (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[voice]" || pip install --no-cache-dir -e .

# Copy source
COPY aria/ aria/

# Data directory for SQLite
RUN mkdir -p /app/data

ENV ARIA_DB_PATH=/app/data/aria.db
ENV ARIA_HOST=0.0.0.0
ENV ARIA_PORT=8000

EXPOSE 8000

CMD ["aria", "serve"]
