FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi[all] \
    uvicorn \
    gradio \
    requests \
    SQLAlchemy \
    python-multipart \
    python-dotenv \
    opencv-python \
    pillow \
    numpy

COPY . .

RUN mkdir -p /app/data/uploads /app/data/output/detection

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]