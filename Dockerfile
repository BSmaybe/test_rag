FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEFAULT_INDEX_DIR=/app/data/index \
    DEFAULT_UPLOAD_DIR=/app/data/uploaded \
    DEFAULT_LOG_DIR=/app/data/logs \
    DEFAULT_LLM_PATH=/app/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY rag_agent /app/rag_agent
COPY streamlit_app.py /app/streamlit_app.py
COPY data /app/data

RUN mkdir -p /app/data/index /app/data/uploaded /app/data/logs /app/models

EXPOSE 8501

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port ${STREAMLIT_SERVER_PORT:-8501}"]
