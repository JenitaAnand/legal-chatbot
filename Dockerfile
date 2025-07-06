# Base image
FROM python:3.10-slim

# Install system libraries required for FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy all your project files to the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI app (Replace server:app with your actual app variable)
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
