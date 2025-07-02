FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_ENV=production

# Expose default port
EXPOSE 8000

# Direct command with 180-second timeout - NO KEEPALIVE
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180 --graceful-timeout 180 --max-requests 100 --access-logfile - --error-logfile -"]
