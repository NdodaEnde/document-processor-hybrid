FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Create an entrypoint script for dynamic port binding
RUN echo '#!/bin/bash\n\
PORT="${PORT:-8000}"\n\
exec gunicorn app:app --bind "0.0.0.0:$PORT"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Expose default port (will be overridden by environment variable if set)
EXPOSE 8000

# Use the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
