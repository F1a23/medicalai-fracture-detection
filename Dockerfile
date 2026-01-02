FROM python:3.10-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend
COPY backend /app/backend

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r /app/backend/requirements.txt

# Environment variables
ENV PORT=7860

EXPOSE 7860

# Run Flask with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "backend.app:app"]
