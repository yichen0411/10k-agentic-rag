FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY starter/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

COPY . .

EXPOSE 8010

CMD ["sh", "-c", "python -m uvicorn chunk_studio.server:app --host 0.0.0.0 --port ${PORT:-8010}"]
