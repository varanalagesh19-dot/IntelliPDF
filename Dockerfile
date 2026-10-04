FROM python:3.11-slim

WORKDIR /app

# System dependencies for PyMuPDF, FAISS, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data dirs
RUN mkdir -p backend/data/uploads backend/data/vector_db

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860

# Hugging Face runs the Dockerfile, and this is the command
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
