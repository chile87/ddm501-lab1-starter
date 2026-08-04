# =============================================================================
# Dockerfile for Movie Rating Prediction API
# DDM501 - Lab 1: First ML Product
# =============================================================================

# -----------------------------------------------------------------------------
# Base image
# -----------------------------------------------------------------------------
FROM python:3.10-slim

# -----------------------------------------------------------------------------
# Runtime environment
#   PYTHONUNBUFFERED     - log lines reach `docker logs` immediately
#   PYTHONDONTWRITEBYTECODE - no .pyc clutter in the image layer
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# -----------------------------------------------------------------------------
# Set working directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Install curl (used by the health check) and build tools (scikit-surprise
# compiles its Cython extensions from source at install time)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Copy and install dependencies
# Done before copying the source so code edits don't invalidate the pip layer.
# -----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Copy application code (see .dockerignore for what is left out)
# -----------------------------------------------------------------------------
COPY . .

# -----------------------------------------------------------------------------
# Run as an unprivileged user
# -----------------------------------------------------------------------------
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# -----------------------------------------------------------------------------
# Expose port
# -----------------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------------
# Health check
#
# /health deliberately answers 200 even when the model failed to load, so that
# the failure is diagnosable over HTTP. The probe therefore inspects the body:
# a container serving requests without a model is NOT healthy.
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health | grep -q '"status":"healthy"' || exit 1

# -----------------------------------------------------------------------------
# Startup command
# -----------------------------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
