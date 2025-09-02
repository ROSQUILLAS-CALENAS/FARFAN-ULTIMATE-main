# syntax=docker/dockerfile:1

# Use Python 3.12 slim (project targets Python 3.12.x and uses virtualenv)
FROM python:3.12-slim

# Predictable Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/venv \
    PATH="/venv/bin:$PATH"

# Minimal system deps; extend only if your wheels require more
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtualenv inside the container
RUN python -m venv "$VIRTUAL_ENV"

# Workdir
WORKDIR /app

# Cache-friendly dependency install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project (local venv is excluded via .dockerignore)
COPY . /app

# Expose port (adjust if needed)
EXPOSE 8000

# Set environment variables
ENV PORT=8000

# Default command: start canonical web server on port 8000
CMD ["python", "canonical_web_server.py", "--port", "8000", "--no-analysis"]
