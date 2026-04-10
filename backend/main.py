"""
Минимальный FastAPI-сервер для подбора размера одежды.

Что делает этот файл:
1) Принимает параметры пользователя (рост, вес, талия).
2) Считает размер по талии (тот же алгоритм, что во фронтенде).
3) Возвращает размер + confidence.
4) Логирует каждый запрос в SQLite (таблица sessions).
"""

from datetime import datetime, timezone
import sqlite3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Имя файла SQLite-базы в папке backend
DB_PATH = "sessions.db"


# Создаем FastAPI-приложение
app = FastAPI(title="FitVector Size API", version="1.0.0")


# Для тестов включаем CORS для всех доменов.
# В продакшене лучше ограничить список origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CalculateRequest(BaseModel):
    """Входные данные для расчета размера."""

    height: float
    weight: float
    waist: float


class CalculateResponse(BaseModel):
    """Ответ API с размером и уверенностью."""

    size: str
    confidence: float
    method: str


# Те же интервалы, что и во фронтенде
SIZE_RANGES = [
    {"size": "S", "min": 60, "max": 76, "include_max": False},
    {"size": "M", "min": 76, "max": 84, "include_max": False},
    {"size": "L", "min": 84, "max": 94, "include_max": False},
    {"size": "XL", "min": 94, "max": 104, "include_max": True},
    {"size": "XXL", "min": 104, "max": 140, "include_max": True},
]


def init_db() -> None:
    """Создает таблицу sessions, если ее еще нет."""
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                height REAL NOT NULL,
                weight REAL NOT NULL,
                waist REAL NOT NULL,
                size TEXT NOT NULL,
                confidence REAL NOT NULL,
                method TEXT NOT NULL
            )
            """
        )
        connection.commit()


def get_size_by_waist(waist: float) -> dict:
    """
    Находит размерный диапазон для заданной талии.
    Если значение не попало ни в один диапазон,
    возвращаем последний (XXL), как и на фронтенде.
    """
    for range_item in SIZE_RANGES:
        in_range = (
            waist >= range_item["min"] and waist <= range_item["max"]
            if range_item["include_max"]
            else waist >= range_item["min"] and waist < range_item["max"]
        )
        if in_range:
            return range_item
    return SIZE_RANGES[-1]


def calculate_confidence(waist: float, range_item: dict) -> float:
    """
    Тот же расчет confidence:
    - 0.9 в центре диапазона
    - 0.6 на границах
    """
    center = (range_item["min"] + range_item["max"]) / 2
    half_range = (range_item["max"] - range_item["min"]) / 2
    if half_range == 0:
        return 0.9

    distance_to_center = abs(waist - center)
    normalized = min(distance_to_center / half_range, 1)
    confidence = 0.9 - normalized * 0.3
    return max(0.6, min(0.9, confidence))


def log_session(
    height: float,
    weight: float,
    waist: float,
    size: str,
    confidence: float,
    method: str,
) -> None:
    """Сохраняет входные и результат расчета в таблицу sessions."""
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (created_at, height, weight, waist, size, confidence, method)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                height,
                weight,
                waist,
                size,
                confidence,
                method,
            ),
        )
        connection.commit()


@app.on_event("startup")
def on_startup() -> None:
    """Инициализация базы при старте приложения."""
    init_db()


@app.get("/health")
def health() -> dict:
    """Простой health-check для Railway/мониторинга."""
    return {"status": "ok"}


@app.post("/calculate", response_model=CalculateResponse)
def calculate(payload: CalculateRequest) -> CalculateResponse:
    """
    Основной endpoint:
    принимает height/weight/waist и возвращает размер.
    Алгоритм базируется на талии (simple method).
    """
    found_range = get_size_by_waist(payload.waist)
    confidence = calculate_confidence(payload.waist, found_range)

    result = CalculateResponse(
        size=found_range["size"],
        confidence=round(confidence, 2),
        method="simple",
    )

    # Логирование запроса в SQLite
    log_session(
        height=payload.height,
        weight=payload.weight,
        waist=payload.waist,
        size=result.size,
        confidence=result.confidence,
        method=result.method,
    )

    return result

