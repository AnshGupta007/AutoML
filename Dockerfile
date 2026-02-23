FROM python:3.11-slim

# Build args
ARG ENVIRONMENT=production
ENV ENVIRONMENT=${ENVIRONMENT}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install the package
RUN pip install --no-cache-dir -e . --no-deps

# Create non-root user
RUN addgroup --system appgroup && adduser --system --group appuser
RUN chown -R appuser:appgroup /app
USER appuser

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models/trained \
    /app/reports /app/artifacts/logs

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Default command: run FastAPI server
CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
