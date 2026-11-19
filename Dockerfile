# Build the web UI
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npx vite build

# API + model (CPU only)
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[api]"
COPY backend/ backend/
COPY configs/ configs/
COPY --from=ui /ui/dist frontend/dist
ENV CHOLEGUARD_MODEL=/app/models/choleguard.onnx
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
