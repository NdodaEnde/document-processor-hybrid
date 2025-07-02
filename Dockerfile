FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_ENV=production

# Enhanced Gunicorn configuration for timeout protection
ENV GUNICORN_TIMEOUT=180
ENV GUNICORN_WORKERS=1
ENV GUNICORN_MAX_REQUESTS=100
ENV GUNICORN_MAX_REQUESTS_JITTER=10
ENV GUNICORN_GRACEFUL_TIMEOUT=180

# Memory optimization for agentic-doc (these should also be set in Render dashboard)
ENV BATCH_SIZE=1
ENV MAX_WORKERS=1
ENV MAX_RETRIES=10
ENV MAX_RETRY_WAIT_TIME=30
ENV PDF_TO_IMAGE_DPI=72
ENV SPLIT_SIZE=5
ENV EXTRACTION_SPLIT_SIZE=25
ENV RETRY_LOGGING_STYLE=log_msg

# Create an enhanced entrypoint script with timeout protection
RUN echo '#!/bin/bash\n\
PORT="${PORT:-8000}"\n\
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-180}"\n\
GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"\n\
GUNICORN_MAX_REQUESTS="${GUNICORN_MAX_REQUESTS:-100}"\n\
GUNICORN_MAX_REQUESTS_JITTER="${GUNICORN_MAX_REQUESTS_JITTER:-10}"\n\
GUNICORN_GRACEFUL_TIMEOUT="${GUNICORN_GRACEFUL_TIMEOUT:-180}"\n\
\n\
echo "🚀 Starting Document Processing Microservice..."\n\
echo "⚙️  Gunicorn timeout: ${GUNICORN_TIMEOUT}s"\n\
echo "👥 Workers: ${GUNICORN_WORKERS}"\n\
echo "🔄 Max requests per worker: ${GUNICORN_MAX_REQUESTS}"\n\
echo "⏱️  Graceful timeout: ${GUNICORN_GRACEFUL_TIMEOUT}s"\n\
echo "🌐 Port: ${PORT}"\n\
\n\
exec gunicorn app:app \\\n\
    --bind "0.0.0.0:$PORT" \\\n\
    --workers "$GUNICORN_WORKERS" \\\n\
    --timeout "$GUNICORN_TIMEOUT" \\\n\
    --graceful-timeout "$GUNICORN_GRACEFUL_TIMEOUT" \\\n\
    --max-requests "$GUNICORN_MAX_REQUESTS" \\\n\
    --max-requests-jitter "$GUNICORN_MAX_REQUESTS_JITTER" \\\n\
    --access-logfile "-" \\\n\
    --error-logfile "-" \\\n\
    --log-level info \\\n\
    --preload' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Expose default port (will be overridden by environment variable if set)
EXPOSE 8000

# Use the enhanced entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
