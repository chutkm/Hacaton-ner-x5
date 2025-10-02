FROM python:3.9-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

COPY api /app/api
COPY src/model /app/src/model
COPY src/inference.py /app/src/inference.py
COPY src/utils.py /app/src/utils.py  # Только если используется в inference
COPY src/metrics.py /app/src/metrics.py  # Только если используется в inference

COPY models/serve /app/models/serve

RUN python -c "from src.inference import predict_entities; predict_entities('test')"

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]