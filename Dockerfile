# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install openenv-core>=0.2.0

# Copy source
COPY . .

# Expose port
EXPOSE 7860

# Run FastAPI with uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]