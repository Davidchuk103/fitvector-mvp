# Базовый образ Python 3.11
FROM python:3.11-slim

# Системные зависимости для mediapipe и opencv-python-headless:
# libGL.so.1    → нужна opencv/mediapipe
# libglib2.0-0  → нужна glib (mediapipe)
# libgomp1      → OpenMP (numpy, mediapipe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория внутри контейнера
WORKDIR /app

# Сначала копируем зависимости, чтобы Docker-кеш работал эффективнее
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код backend
COPY . .

# Railway обычно прокидывает порт через переменную PORT
ENV PORT=8000

# Запуск FastAPI через uvicorn
# main:app -> файл main.py и объект app внутри него
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

