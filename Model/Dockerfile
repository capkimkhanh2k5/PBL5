# ============================================
# Python Flask + PyTorch AI Model Service
# ============================================
FROM python:3.11-slim

# System dependencies for OpenCV + cmake for onnxsim build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    cmake \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Security: non-root user
RUN addgroup --gid 1001 appgroup && \
    adduser --uid 1001 --gid 1001 --disabled-password --gecos "" appuser

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy model and application files
COPY data.yaml .
COPY trainModel/model.pth trainModel/model.pth
COPY testCam.py .

# Set ownership
RUN chown -R appuser:appgroup /app

USER 1001
EXPOSE 9999

# Environment variables
ENV FLASK_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9999/ || exit 1

CMD ["python", "testCam.py"]
