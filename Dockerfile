# Use a lightweight Python base
FROM python:3.11-slim

# Install system packages for LlamaCpp & Chroma
RUN apt-get update && apt-get install -y \
    build-essential cmake curl libssl-dev git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
