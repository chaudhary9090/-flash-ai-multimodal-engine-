# ==============================================================================
# Multi-Stage Production Dockerfile for Custom GPT Platform
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Dependencies
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only wheel explicitly to drastically shrink image footprint
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Image
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS final

WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user for security compliance
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application source & frontend assets
COPY app/ ./app/
COPY frontend/ ./frontend/

USER appuser

EXPOSE 8000

# Container Healthcheck using API v1 health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
