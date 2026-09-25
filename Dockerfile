# Single image for HuggingFace Spaces (port 7860) and Azure (honors $PORT).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Deps first so code edits don't bust the pip layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# stdlib healthcheck (slim has no curl); hits the port the app actually binds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/').read()" || exit 1

# main() reads PORT (default 7860) — one place decides the port for every target.
CMD ["python", "-m", "server.app"]
