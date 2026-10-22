FROM python:3.12-slim

WORKDIR /app

# OpenCV and MediaPipe need these shared libraries on slim images
# (libegl1 / libgles2 provide libEGL.so.1 / libGLESv2.so.2 that mediapipe loads at import)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
