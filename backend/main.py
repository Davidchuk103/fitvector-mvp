"""
Минимальный FastAPI-сервер для подбора размера одежды.

Что делает этот файл:
1) Принимает параметры пользователя (рост, вес, талия).
2) Считает размер по талии (тот же алгоритм, что во фронтенде).
3) Возвращает размер + confidence.
4) Сохраняет сессии в Supabase (таблица sessions) и при сбое — в SQLite.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Загружаем переменные из .env:
# - сначала backend/.env (если есть)
# - затем корень репозитория ../.env (как у тебя в fitvector-mvp/.env)
_backend_dir = Path(__file__).resolve().parent
_repo_root = _backend_dir.parent
load_dotenv(_backend_dir / ".env")
load_dotenv(_repo_root / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Имя файла SQLite-базы в папке backend (fallback)
DB_PATH = str(_backend_dir / "sessions.db")

# --- Supabase: где взять URL и ключ ---
# Dashboard → Project Settings → API:
#   Project URL  → SUPABASE_URL
#   anon public / service_role — в переменную SUPABASE_KEY (как в задании)
#   или отдельно SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY (удобно для .env)
# Для INSERT с сервера без настройки RLS чаще используют service_role (никогда не светить во фронт).
_supabase_client: Optional[Any] = None
_supabase_tried_init = False


def _resolve_supabase_key() -> Optional[str]:
    return (
        os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )


def get_supabase_client() -> Optional[Any]:
    """
    Ленивая инициализация клиента supabase-py.
    Если URL/ключ не заданы — None (работаем только с SQLite).
    """
    global _supabase_client, _supabase_tried_init

    if _supabase_tried_init:
        return _supabase_client

    _supabase_tried_init = True
    url = os.getenv("SUPABASE_URL")
    key = _resolve_supabase_key()
    if not url or not key:
        logger.info("Supabase: SUPABASE_URL или ключ не заданы — только SQLite.")
        _supabase_client = None
        return None

    try:
        from supabase import create_client

        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supabase: не удалось создать клиент: %s", exc)
        _supabase_client = None
        return None


app = FastAPI(title="FitVector Size API", version="1.1.0")

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


class StatsResponse(BaseModel):
    """Сводка по сохранённым сессиям."""

    total_sessions: int
    # S/M/L/XL по заданию; XXL добавлен, чтобы не терять записи сверх XL
    size_distribution: dict[str, int]
    avg_confidence: float


SIZE_RANGES = [
    {"size": "S", "min": 60, "max": 76, "include_max": False},
    {"size": "M", "min": 76, "max": 84, "include_max": False},
    {"size": "L", "min": 84, "max": 94, "include_max": False},
    {"size": "XL", "min": 94, "max": 104, "include_max": True},
    {"size": "XXL", "min": 104, "max": 140, "include_max": True},
]

# Ключи для ответа /stats (S–XL по заданию; XXL учитываем отдельно в словаре)
STATS_SIZE_KEYS = ("S", "M", "L", "XL", "XXL")


def init_db() -> None:
    """Создает локальную таблицу sessions (fallback), если её ещё нет."""
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
    """Находит размерный диапазон для заданной талии."""
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
    """Уверенность: 0.9 в центре диапазона, 0.6 на границах."""
    center = (range_item["min"] + range_item["max"]) / 2
    half_range = (range_item["max"] - range_item["min"]) / 2
    if half_range == 0:
        return 0.9

    distance_to_center = abs(waist - center)
    normalized = min(distance_to_center / half_range, 1)
    confidence = 0.9 - normalized * 0.3
    return max(0.6, min(0.9, confidence))


def log_session_sqlite(
    height: float,
    weight: float,
    waist: float,
    size: str,
    confidence: float,
    method: str,
) -> None:
    """Fallback: запись в локальный SQLite."""
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


def persist_session(
    height: float,
    weight: float,
    waist: float,
    size: str,
    confidence: float,
    method: str,
) -> None:
    """
    Сначала пишем в Supabase (схема Table Editor: height, weight, waist, size, confidence).
    Если Supabase недоступен — дублируем в SQLite.
    """
    client = get_supabase_client()
    row = {
        "height": int(round(height)),
        "weight": int(round(weight)),
        "waist": int(round(waist)),
        "size": size,
        "confidence": float(confidence),
    }

    if client is not None:
        try:
            client.table("sessions").insert(row).execute()
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supabase insert не удался, пишем в SQLite: %s", exc)

    log_session_sqlite(height, weight, waist, size, confidence, method)


def _stats_from_sqlite() -> StatsResponse:
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*), AVG(confidence) FROM sessions")
        total, avg_c = cursor.fetchone()
        total = int(total or 0)
        avg_confidence = float(avg_c) if avg_c is not None else 0.0

        dist = {k: 0 for k in STATS_SIZE_KEYS}
        cursor.execute("SELECT size, COUNT(*) FROM sessions GROUP BY size")
        for size, cnt in cursor.fetchall():
            if size in dist:
                dist[size] = int(cnt)

    return StatsResponse(
        total_sessions=total,
        size_distribution={k: dist[k] for k in STATS_SIZE_KEYS},
        avg_confidence=round(avg_confidence, 4) if total else 0.0,
    )


def _stats_from_supabase(client: Any) -> StatsResponse:
    res = client.table("sessions").select("size,confidence").execute()
    rows = res.data or []
    total = len(rows)
    if total == 0:
        return StatsResponse(
            total_sessions=0,
            size_distribution={k: 0 for k in STATS_SIZE_KEYS},
            avg_confidence=0.0,
        )

    dist = {k: 0 for k in STATS_SIZE_KEYS}
    conf_sum = 0.0
    for row in rows:
        s = row.get("size")
        if s in dist:
            dist[s] += 1
        c = row.get("confidence")
        if c is not None:
            conf_sum += float(c)

    avg_confidence = conf_sum / total
    return StatsResponse(
        total_sessions=total,
        size_distribution={k: dist[k] for k in STATS_SIZE_KEYS},
        avg_confidence=round(avg_confidence, 4),
    )


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    """
    Сводка по таблице sessions в Supabase.
    При ошибке или отсутствии ключей — расчёт по локальному SQLite.
    """
    client = get_supabase_client()
    if client is not None:
        try:
            return _stats_from_supabase(client)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supabase /stats fallback на SQLite: %s", exc)

    try:
        return _stats_from_sqlite()
    except Exception as exc:  # noqa: BLE001
        logger.exception("SQLite stats failed: %s", exc)
        raise


@app.post("/calculate", response_model=CalculateResponse)
def calculate(payload: CalculateRequest) -> CalculateResponse:
    found_range = get_size_by_waist(payload.waist)
    confidence = calculate_confidence(payload.waist, found_range)

    result = CalculateResponse(
        size=found_range["size"],
        confidence=round(confidence, 2),
        method="simple",
    )

    persist_session(
        height=payload.height,
        weight=payload.weight,
        waist=payload.waist,
        size=result.size,
        confidence=result.confidence,
        method=result.method,
    )

    return result
