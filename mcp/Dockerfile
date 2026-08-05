FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8080

RUN pip install --no-cache-dir "mcp>=1.2.0" "httpx>=0.27.0" "uvicorn>=0.30.0"

COPY server.py ./server.py

EXPOSE 8080

CMD ["python", "server.py"]
