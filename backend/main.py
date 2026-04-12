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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from measure import (
    calculate_measurements,
    get_confidence,
    validate_all,
)

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

# --- Supabase (supabase-py / pip install supabase) ---
# Dashboard → Project Settings → API:
#   Project URL → SUPABASE_URL
#   Ключ API   → SUPABASE_KEY (часто подставляют service_role для бэкенда)
#   Альтернатива: SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY
_supabase_client: Optional[Any] = None
_supabase_tried_init = False


def _resolve_supabase_key() -> Optional[str]:
    """
    Ключ API: в задании — SUPABASE_KEY (часто кладут service_role для бэкенда).
    Дополнительно поддерживаем старые имена переменных.
    """
    return (
        os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )


def get_supabase_client() -> Optional[Any]:
    """
    Ленивая инициализация клиента supabase-py (пакет pip: supabase).
    Нужны SUPABASE_URL и SUPABASE_KEY (см. Project Settings → API).
    Если не заданы — None (только SQLite).
    """
    global _supabase_client, _supabase_tried_init

    if _supabase_tried_init:
        return _supabase_client

    _supabase_tried_init = True
    url = os.getenv("SUPABASE_URL")
    key = _resolve_supabase_key()
    if not url or not key:
        logger.info(
            "Supabase: задайте SUPABASE_URL и SUPABASE_KEY — иначе только SQLite."
        )
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


app = FastAPI(title="FitVector Size API", version="1.2.0")

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
    # Только S, M, L, XL — размер XXL в ответе суммируется в XL
    size_distribution: dict[str, int]
    avg_confidence: float


SIZE_RANGES = [
    {"size": "S", "min": 60, "max": 76, "include_max": False},
    {"size": "M", "min": 76, "max": 84, "include_max": False},
    {"size": "L", "min": 84, "max": 94, "include_max": False},
    {"size": "XL", "min": 94, "max": 104, "include_max": True},
    {"size": "XXL", "min": 104, "max": 140, "include_max": True},
]

# Ответ /stats: только S, M, L, XL (XXL включаем в счётчик XL)
STATS_SIZE_KEYS = ("S", "M", "L", "XL")


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


def _merge_size_distribution(raw: dict[str, int]) -> dict[str, int]:
    """Приводит распределение к ключам S, M, L, XL (XXL → XL)."""
    return {
        "S": int(raw.get("S", 0)),
        "M": int(raw.get("M", 0)),
        "L": int(raw.get("L", 0)),
        "XL": int(raw.get("XL", 0)) + int(raw.get("XXL", 0)),
    }


def _stats_from_sqlite() -> StatsResponse:
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*), AVG(confidence) FROM sessions")
        total, avg_c = cursor.fetchone()
        total = int(total or 0)
        avg_confidence = float(avg_c) if avg_c is not None else 0.0

        dist_full: dict[str, int] = {"S": 0, "M": 0, "L": 0, "XL": 0, "XXL": 0}
        cursor.execute("SELECT size, COUNT(*) FROM sessions GROUP BY size")
        for size, cnt in cursor.fetchall():
            if size in dist_full:
                dist_full[size] = int(cnt)

    return StatsResponse(
        total_sessions=total,
        size_distribution=_merge_size_distribution(dist_full),
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

    dist_full = {"S": 0, "M": 0, "L": 0, "XL": 0, "XXL": 0}
    conf_sum = 0.0
    for row in rows:
        s = row.get("size")
        if s in dist_full:
            dist_full[s] += 1
        c = row.get("confidence")
        if c is not None:
            conf_sum += float(c)

    avg_confidence = conf_sum / total
    return StatsResponse(
        total_sessions=total,
        size_distribution=_merge_size_distribution(dist_full),
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


class MeasureResponse(BaseModel):
    height_cm: float
    weight_kg: Optional[float] = None
    gender: Optional[str] = None
    cv_detected: bool
    confidence: float
    pose_score: float
    pose_issues: list[str]
    debug_info: Optional[dict] = None
    measurements: dict[str, dict]


@app.post("/measure", response_model=MeasureResponse)
async def measure(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    height_cm: float = Form(...),
    weight_kg: Optional[float] = Form(None),
    gender: Optional[str] = Form("unknown"),
):
    front_bytes = await front.read()
    side_bytes = await side.read()

    validation = validate_all(front_bytes, side_bytes)
    if not validation.valid:
        raise HTTPException(status_code=400, detail=validation.errors)

    measurements, cv_detected, extra_info = calculate_measurements(
        height_cm=height_cm,
        weight_kg=weight_kg,
        front_bytes=front_bytes,
        side_bytes=side_bytes,
        gender=gender or "unknown",
    )

    confidence = get_confidence(
        cv_detected=cv_detected,
        has_weight=weight_kg is not None,
        pose_score=extra_info.get("pose_score", 0.0),
        visibility_ratio=extra_info.get("front_visibility", 0.0),
        gender_known=gender in ("male", "female"),
    )

    return MeasureResponse(
        height_cm=height_cm,
        weight_kg=weight_kg,
        gender=gender,
        cv_detected=cv_detected,
        confidence=round(confidence, 2),
        pose_score=extra_info.get("pose_score", 0.0),
        pose_issues=extra_info.get("pose_issues", []),
        debug_info=extra_info.get("debug_info"),
        measurements=measurements,
    )
