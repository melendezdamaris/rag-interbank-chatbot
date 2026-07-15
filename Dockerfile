FROM python:3.10-slim

WORKDIR /app

# Copiar dependencias primero (mejor cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Correr indexer al iniciar y luego levantar la API
CMD python src/indexer.py && uvicorn src.api:app --host 0.0.0.0 --port $PORT