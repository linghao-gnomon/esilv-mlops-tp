FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/

# Copy data directory (needed for training)
COPY data/ data/

# Train the model during build
RUN python src/train.py --n_trees 20

# Expose API port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
