FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY api/ ./api/

# Fetch the weights from the GitHub Release at build time, so the container starts ready to serve
RUN python -c "from src.prediction import ensure_weights; ensure_weights()"

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]