"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import store
from .config import settings
from .contracts import HealthResponse
from .routes import health, router
from .vision.classifier import get_classifier, missing_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("foshol_lens")

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await store.init_db()
    purged = await store.purge_older_than(settings.analysis_retention_days)
    if purged:
        log.info("purged %d analyses older than %d days", purged, settings.analysis_retention_days)
    if not settings.llm_configured:
        llm_status = "template fallback"
    elif settings.llm_provider == "openai":
        llm_status = f"openai ({settings.llm_model})"
    else:
        llm_status = f"gemini ({settings.gemini_model})"
    log.info("LLM: %s", llm_status)
    log.info("TTS: %s", "enabled" if settings.tts_enabled else "disabled")
    _log_vision_status()
    yield


def _log_vision_status() -> None:
    """Say plainly at boot whether vision can run.

    Without this, a missing checkpoint is invisible until a farmer uploads a
    photo — and then it looks like the model failed rather than never ran.
    """
    if not settings.vision_enabled:
        log.info("Vision: disabled")
        return

    classifier = get_classifier()
    spec = classifier.spec
    missing = missing_weights(spec)
    if missing is None:
        log.info("Vision: enabled (%s)", spec["id"])
        return

    log.warning(
        "Vision: enabled (%s) but the checkpoint is missing at %s — every photo "
        "will be reported as unidentified. Put %s there, or set VISION_MODEL_DIR.",
        spec["id"],
        missing,
        spec["file"],
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="ফসল লেন্স API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # A dev server that drifts off port 5173 still has to work: Vite silently
    # picks the next free port, and rejecting that surfaces as an upload failure
    # that looks like a broken server rather than a CORS problem.
    dev_origin_regex = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=dev_origin_regex if settings.is_development else None,
        allow_credentials="*" not in settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
    app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()
