FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV CHROMA_PATH=/app/data/chroma_db
ENV REPO_PATH=/app
ENV MCP_TRANSPORT=sse

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python dependencies
COPY requirements.txt .
# We need to make sure uvicorn/starlette (sse dependencies) are installed 
# fastmcp[sse] installs them in FastMCP v3, but we can also just rely on FastMCP 
# installing its own dependencies if we already have it in requirements.txt. 
# However, requirements.txt has 'fastmcp>=3.0.0'. Let's ensure uvicorn and starlette are available:
RUN pip install --no-cache-dir -r requirements.txt fastmcp[sse] uvicorn starlette sse-starlette

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application
COPY . .

# Ensure data directory exists
RUN mkdir -p /app/data/chroma_db

# Expose the port
EXPOSE 8000

# Start the FastMCP server
CMD ["python", "src/server.py"]
