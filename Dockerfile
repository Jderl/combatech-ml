FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY original_files_ml ./original_files_ml

ENV PYTHONPATH=/app/src

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "combatech_ml.api.main:app", "--host", "0.0.0.0", "--port", "8001"]