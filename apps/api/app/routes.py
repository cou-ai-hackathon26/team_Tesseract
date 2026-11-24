"""HTTP routes.

Thin: parse, delegate, serialise. All pipeline logic lives in `jobs.pipeline`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from . import store
from .config import settings
from .contracts import (
    AnalysisResult,
    AnalysisSummary,
    CreateAnalysisResponse,
    District,
    HealthResponse,
    TtsRequest,
    Weather,
)
from .grounding import knowledge_base
from .grounding.districts import DISTRICTS, find_nearest_district, get_district
from .grounding.weather import fetch_weather
from .jobs import pipeline
from .tts.engine import (
    TtsSafetyError,
    TtsUnavailable,
    audio_mime,
    get_engine,
    sanitize_for_speech,
)
from .vision.classifier import get_classifier

log = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_TTS_TEXT_CHARS = 600
TTS_MEMORY_CACHE_MAX = 64
_tts_memory_cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()


def clear_tts_memory_cache() -> None:
    """Clear the bounded in-memory on-demand TTS cache (used in tests)."""
    _tts_memory_cache.clear()
ALLOWED_AUDIO_MIMES = frozenset(
    {
        "audio/webm",
        "audio/ogg",
        "audio/mp4",
        "audio/mpeg",
        "audio/wav",
        "audio/aac",
        "audio/x-m4a",
        "audio/x-wav",
        "video/mp4",
        "video/webm",
    }
)
_AUDIO_MIME_ALIASES: dict[str, str] = {
    "audio/x-m4a": "audio/mp4",
    "audio/x-wav": "audio/wav",
}
JPEG_QUALITY = 82
MAX_IMAGE_EDGE = 1280

VERSION = "0.1.0"


def _normalize_audio_mime(content_type: str | None) -> str:
    """Strip codec parameters such as `;codecs=opus`, normalise case, and alias."""
    if not content_type:
        return ""
    base = content_type.split(";", 1)[0].strip().lower()
    return _AUDIO_MIME_ALIASES.get(base, base)


def _clean_question_text(value: str | None) -> str | None:
    if not value:
        return None
    collapsed = " ".join(value.split()).strip()
    if not collapsed:
        return None
    return collapsed[: settings.max_question_chars].strip() or None


def _summary_from_row(row: dict[str, Any]) -> AnalysisSummary:
    return AnalysisSummary(
        id=row["id"],
        createdAt=row["created_at"],
        cropName=_crop_name(row["crop_id"]),
        status=row["status"],
        stage=row["stage"],
        thumbnailUrl=f"/api/v1/analyses/{row['id']}/image",
        imageUrl=f"/api/v1/analyses/{row['id']}/image",
        diseaseLabel=_disease_label(row.get("disease_id")),
        district=_district_name(row.get("district")),
    )


def _district_name(district_id: str | None) -> str | None:
    """The id is stored; the UI needs the Bangla name."""
    if not district_id:
        return None
    district = get_district(district_id)
    return str(district["name"]) if district else None


def _crop_name(crop_id: str) -> str:
    for crop in knowledge_base.crops():
        if crop["id"] == crop_id:
            return crop["name"]
    return "শনাক্ত করা যায়নি"


def _disease_label(disease_id: str | None) -> str | None:
    if not disease_id:
        return None
    entry = knowledge_base.get_disease(disease_id)
    return entry["diseaseLabel"] if entry else None


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engine = get_engine()
    return HealthResponse(
        status="ok" if settings.llm_configured else "degraded",
        llm="ready" if settings.llm_configured else "unconfigured",
        tts="ready" if engine.available else "unavailable",
        # Asks the classifier, not the env flag: a local checkpoint that is not
        # on disk reports "unavailable" instead of claiming to be ready and then
        # failing every upload silently.
        vision=get_classifier().status,
        weather="ready",
        db="ok",
        version=VERSION,
    )


@router.get("/districts", response_model=list[District])
async def districts() -> list[District]:
    return [District(**district) for district in DISTRICTS]  # type: ignore[arg-type]


@router.get("/weather", response_model=Weather)
async def weather_lookup(
    lat: float,
    lon: float,
    label: str | None = None,
) -> Weather:
    if not (
        math.isfinite(lat)
        and math.isfinite(lon)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
    ):
        raise HTTPException(status_code=400, detail="invalid_coordinates")

    nearest = find_nearest_district(lat, lon)
    cleaned_label = " ".join((label or "").split()).strip()[:120] or None
    facts = await fetch_weather(
        latitude=lat,
        longitude=lon,
        district_id=str(nearest["id"]),
        district_name=str(nearest["name"]),
        location_source="voice" if cleaned_label else "gps",
        resolved_place=cleaned_label or str(nearest["name"]),
    )
    if facts is None:
        raise HTTPException(status_code=503, detail="weather_unavailable")
    return Weather.model_validate(facts.as_contract_dict())


@router.get("/crops")
async def crops() -> list[dict[str, Any]]:
    return [
        {
            "id": crop["id"],
            "name": crop["name"],
            "diseases": [disease["diseaseLabel"] for disease in crop["diseases"]],
        }
        for crop in knowledge_base.crops()
    ]


@router.post("/analyses", status_code=202, response_model=CreateAnalysisResponse)
async def create_analysis(
    image: Annotated[UploadFile, File()],
    crop: Annotated[str, Form()] = "rice",
    district: Annotated[str | None, Form()] = None,
    latitude: Annotated[float | None, Form()] = None,
    longitude: Annotated[float | None, Form()] = None,
    question_text: Annotated[str | None, Form()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
) -> CreateAnalysisResponse:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty_upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image_too_large")

    district = (district or "").strip() or None
    if district is not None and get_district(district) is None:
        raise HTTPException(status_code=400, detail="unknown_district")

    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=400, detail="invalid_coordinates")
    if latitude is not None and longitude is not None and not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    ):
        raise HTTPException(status_code=400, detail="invalid_coordinates")

    # Rice and potato only: those are the crops the shipped model can name, and
    # the only ones with curated advice. Accepting tomato or chili here would
    # store a row that can only ever answer "could not identify".
    if crop not in knowledge_base.SUPPORTED_CROPS:
        raise HTTPException(status_code=400, detail="unsupported_crop")

    cleaned_question = _clean_question_text(question_text)

    audio_bytes: bytes | None = None
    audio_mime: str | None = None
    if audio is not None:
        raw_audio = await audio.read()
        if raw_audio:
            if len(raw_audio) > settings.max_audio_upload_bytes:
                raise HTTPException(status_code=413, detail="audio_too_large")
            normalized_mime = _normalize_audio_mime(audio.content_type)
            if normalized_mime not in ALLOWED_AUDIO_MIMES:
                raise HTTPException(status_code=400, detail="unsupported_audio_type")
            audio_bytes = raw_audio
            audio_mime = normalized_mime

    analysis_id = str(uuid.uuid4())
    created_at = datetime.now(UTC).isoformat()

    await store.insert_analysis(
        analysis_id=analysis_id,
        created_at=created_at,
        crop_id=crop,
        district=district,
    )

    # Store the image now; the job only needs the id.
    destination = settings.images_dir / f"{analysis_id}.jpg"
    try:
        _write_image(raw, destination)
    except Exception as exc:
        await store.mark_failed(analysis_id, "image_decode_failed")
        raise HTTPException(status_code=400, detail="image_unreadable") from exc

    # Fire and forget: the client polls status.
    pipeline.spawn(
        analysis_id,
        crop,
        district,
        latitude=latitude,
        longitude=longitude,
        question_text=cleaned_question,
        audio_bytes=audio_bytes,
        audio_mime=audio_mime,
    )

    return CreateAnalysisResponse(
        id=analysis_id,
        status="pending",
        stage="queued",
        statusUrl=f"/api/v1/analyses/{analysis_id}",
    )


def _write_image(raw: bytes, destination: Any) -> None:
    """Downscale, re-encode, and drop EXIF.

    EXIF is dropped for a specific reason: phones write GPS coordinates into
    photos, and a farmer's location should not land on our disk because their
    camera helpfully recorded it.
    """
    import io

    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=JPEG_QUALITY, optimize=True)


@router.get("/analyses", response_model=list[AnalysisSummary])
async def list_analyses(limit: int = 50) -> list[AnalysisSummary]:
    rows = await store.list_rows(limit=min(max(limit, 1), 200))
    return [_summary_from_row(row) for row in rows]


@router.get("/analyses/{analysis_id}", response_model=AnalysisSummary)
async def get_analysis(analysis_id: str) -> AnalysisSummary:
    row = await store.get_row(analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="analysis_not_found")
    return _summary_from_row(row)


@router.get("/analyses/{analysis_id}/result", response_model=AnalysisResult)
async def get_result(analysis_id: str) -> AnalysisResult:
    row = await store.get_row(analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="analysis_not_found")
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"not_ready:{row['stage']}")
    return AnalysisResult.model_validate(json.loads(row["result_json"]))


@router.get("/analyses/{analysis_id}/image")
async def get_image(analysis_id: str) -> FileResponse:
    path = settings.images_dir / f"{analysis_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="image_not_found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/analyses/{analysis_id}/audio", response_model=None)
async def get_audio(analysis_id: str) -> Any:
    row = await store.get_row(analysis_id)
    if row is None or not row.get("result_json"):
        raise HTTPException(status_code=404, detail="analysis_not_found")

    audio = json.loads(row["result_json"]).get("audio") or {}
    if audio.get("status") != "ready":
        # 202 means "keep polling", not "error".
        return JSONResponse({"status": audio.get("status", "none")}, status_code=202)

    filename = audio.get("file")
    path = settings.audio_dir / filename if filename else None
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="audio_not_found")

    media_type = audio_mime(path)
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.post("/tts", response_model=None)
async def synthesize_tts(payload: TtsRequest) -> Response:
    cleaned = " ".join(payload.text.split()).strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="empty_text")
    if len(cleaned) > MAX_TTS_TEXT_CHARS:
        raise HTTPException(status_code=400, detail="text_too_long")

    speakable = sanitize_for_speech(cleaned)
    if not speakable:
        raise HTTPException(status_code=400, detail="empty_text")

    engine = get_engine()
    if not engine.available:
        raise HTTPException(status_code=503, detail="tts_unavailable")

    cache_key = hashlib.sha256(
        f"{speakable}|{settings.tts_voice}|{settings.tts_format}".encode()
    ).hexdigest()
    cached = _tts_memory_cache.get(cache_key)
    if cached is not None:
        _tts_memory_cache.move_to_end(cache_key)
        audio_bytes, mime_type = cached
        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    try:
        clip = await engine.synthesise(speakable, settings.audio_dir)
    except TtsSafetyError as exc:
        raise HTTPException(status_code=400, detail="tts_safety_rejected") from exc
    except TtsUnavailable as exc:
        raise HTTPException(status_code=503, detail="tts_unavailable") from exc
    except Exception as exc:
        log.warning("on-demand TTS failed: %s: %s", exc.__class__.__name__, str(exc) or "no detail")
        raise HTTPException(status_code=503, detail="tts_failed") from exc

    audio_bytes = clip.path.read_bytes()
    mime_type = clip.mime
    _tts_memory_cache[cache_key] = (audio_bytes, mime_type)
    if len(_tts_memory_cache) > TTS_MEMORY_CACHE_MAX:
        _tts_memory_cache.popitem(last=False)

    return Response(
        content=audio_bytes,
        media_type=mime_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.delete("/analyses/{analysis_id}", status_code=204)
async def delete_analysis(analysis_id: str) -> None:
    row = await store.get_row(analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="analysis_not_found")
    (settings.images_dir / f"{analysis_id}.jpg").unlink(missing_ok=True)
    for suffix in ("opus", "mp3", "wav"):
        (settings.audio_dir / f"{analysis_id}.{suffix}").unlink(missing_ok=True)
    async with store.connect() as db:
        await db.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        await db.commit()


def _disk_free_mb(path: Any) -> int:
    return int(shutil.disk_usage(path).free / (1024 * 1024))
