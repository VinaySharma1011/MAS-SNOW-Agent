

FROM python:3.11-slim

# System dependencies for Azure SDK, cryptography, and ServiceNow integration
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker cache utilization
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose FastAPI port
EXPOSE 8080

# Default environment variables (override in production)
ENV PORT=8080

# Healthcheck (optional, for container orchestrators)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:8080/health || exit 1

# Entrypoint
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8080"]

