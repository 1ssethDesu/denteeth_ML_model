# Use Python 3.9 slim image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies, including curl
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the model file (if it exists)
COPY model/faster_rcnn_model.pth ./model/faster_rcnn_model.pth

# Copy application source code
COPY app/ ./app/

# Copy supporting files
COPY run.py .
COPY pytest.ini .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose the port
EXPOSE 8000

# Health check with a longer start period
HEALTHCHECK --interval=30s --timeout=30s --start-period=2m --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run the application
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]