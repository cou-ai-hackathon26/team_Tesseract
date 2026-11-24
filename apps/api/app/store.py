"""SQLite persistence.

One file, no server, real transactions. WAL mode so a reading request never
blocks the job writer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

from .config import settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    crop_id      TEXT NOT NULL,
    disease_id   TEXT,
    status       TEXT NOT NULL,
    stage        TEXT NOT NULL,
    district     TEXT,
    generated_by TEXT,
    result_json  TEXT,
    symptoms_text TEXT,
    treatment_text TEXT,
    prevention_text TEXT,
    narrative    TEXT,
    error_code   TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC);
"""


@asynccontextmanager
async def connect() -> AsyncIterator[aiosqlite.Connection]:
    connection = await aiosqlite.connect(settings.database_path)
    connection.row_factory = aiosqlite.Row
    try:
        yield connection
    finally:
        await connection.close()


async def init_db() -> None:
    async with connect() as db:
        await db.executescript(SCHEMA)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.commit()


async def insert_analysis(
    *,
    analysis_id: str,
    created_at: str,
    crop_id: str,
    district: str | None,
) -> None:
    async with connect() as db:
        await db.execute(
            "INSERT INTO analyses (id, created_at, crop_id, status, stage, district)"
            " VALUES (?, ?, ?, 'pending', 'queued', ?)",
            (analysis_id, created_at, crop_id, district),
        )
        await db.commit()


async def update_stage(analysis_id: str, stage: str, status: str | None = None) -> None:
    """Move the row forward.

    The stage is mirrored into `result_json` when one exists, because that is
    what `GET .../result` serves - a result that still claims `speaking` after
    the audio is written would keep the client polling forever.
    """
    async with connect() as db:
        if status:
            await db.execute(
                "UPDATE analyses SET stage = ?, status = ? WHERE id = ?",
                (stage, status, analysis_id),
            )
        else:
            await db.execute("UPDATE analyses SET stage = ? WHERE id = ?", (stage, analysis_id))

        cursor = await db.execute("SELECT result_json FROM analyses WHERE id = ?", (analysis_id,))
        row = await cursor.fetchone()
        if row is not None and row["result_json"]:
            result = json.loads(row["result_json"])
            if result.get("stage") != stage:
                result["stage"] = stage
                await db.execute(
                    "UPDATE analyses SET result_json = ? WHERE id = ?",
                    (json.dumps(result, ensure_ascii=False), analysis_id),
                )
        await db.commit()


async def save_result(analysis_id: str, result: dict[str, Any], *, disease_id: str) -> None:
    async with connect() as db:
        await db.execute(
            """
            UPDATE analyses SET
                status = ?, stage = ?, disease_id = ?, generated_by = ?,
                result_json = ?, narrative = ?,
                symptoms_text = ?, treatment_text = ?, prevention_text = ?
            WHERE id = ?
            """,
            (
                result["status"],
                result["stage"],
                disease_id,
                result.get("generatedBy"),
                json.dumps(result, ensure_ascii=False),
                result.get("narrative", ""),
                json.dumps(result.get("symptoms", []), ensure_ascii=False),
                json.dumps(result.get("treatment"), ensure_ascii=False),
                json.dumps(result.get("prevention", []), ensure_ascii=False),
                analysis_id,
            ),
        )
        await db.commit()


async def update_audio(analysis_id: str, audio: dict[str, Any]) -> None:
    """Patch just the audio block, after text has already been served."""
    async with connect() as db:
        cursor = await db.execute("SELECT result_json FROM analyses WHERE id = ?", (analysis_id,))
        row = await cursor.fetchone()
        if row is None or row["result_json"] is None:
            return
        result = json.loads(row["result_json"])
        result["audio"] = audio
        await db.execute(
            "UPDATE analyses SET result_json = ? WHERE id = ?",
            (json.dumps(result, ensure_ascii=False), analysis_id),
        )
        await db.commit()


async def mark_failed(analysis_id: str, error_code: str) -> None:
    async with connect() as db:
        await db.execute(
            "UPDATE analyses SET status = 'failed', stage = 'done', error_code = ? WHERE id = ?",
            (error_code, analysis_id),
        )
        await db.commit()


async def get_row(analysis_id: str) -> dict[str, Any] | None:
    async with connect() as db:
        cursor = await db.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def list_rows(limit: int = 100) -> list[dict[str, Any]]:
    async with connect() as db:
        cursor = await db.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in await cursor.fetchall()]


async def purge_older_than(days: int) -> int:
    async with connect() as db:
        cursor = await db.execute(
            "DELETE FROM analyses WHERE created_at < datetime('now', ?)",
            (f"-{days} days",),
        )
        await db.commit()
        return cursor.rowcount or 0
