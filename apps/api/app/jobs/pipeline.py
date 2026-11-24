"""The analysis job.

Runs stages in order, persisting progress so `GET /analyses/{id}` can report it.
Each stage degrades independently: a weather timeout, a vision miss, or a TTS
failure never withholds the diagnosis or the curated treatment.

Vision reports; it does not decide. The classifier can name a leaf, decline to
name it, or be unavailable, and each of those three has an honest representation
downstream. What it must never do is supply a confidence nobody computed, which
is why the `uncertain` flag and the confidence the LLM sees are derived from the
classifier's actual output rather than assumed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from .. import store
from ..config import settings
from ..grounding import knowledge_base, rag
from ..grounding.districts import find_nearest_district, get_district
from ..grounding.geocoder import resolve_precise_location
from ..grounding.weather import build_extension_office_bn, fetch_weather
from ..llm.audio_input import (
    QuestionContext,
    extract_question_context,
    extract_question_context_offline,
    transcribe_bangla_question,
)
from ..llm.writer import InvalidNarrative, get_writer
from ..tts.engine import TtsSafetyError, get_engine
from ..vision import VisionFacts, get_classifier

log = logging.getLogger(__name__)
_DEFAULT_TRANSCRIBE = transcribe_bangla_question

_CROP_BN_MAP: dict[str, str] = {
    "rice": "ধান",
    "potato": "আলু",
}

#: Which KB entry is used when the classifier cannot name the disease. Every
#: crop still gets real, curated treatment; the result says it is unconfirmed.
#: Rice and potato only: those are the two crops the shipped model can name.
FALLBACK_DISEASE = {
    "rice": "rice_blast",
    "potato": "potato_late_blight",
}

#: Confidence band boundaries, applied to the classifier's probability.
HIGH_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.45

CONCURRENCY = 2

#: Created per event loop. A module-level `Semaphore` binds to the loop that
#: first awaits it, so a second loop (uvicorn reload, pytest) would raise
#: "bound to a different event loop" instead of queueing.
_semaphores: dict[int, asyncio.Semaphore] = {}


def _semaphore_for_current_loop() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _semaphores.get(id(loop))
    if semaphore is None:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        _semaphores[id(loop)] = semaphore
    return semaphore


#: Strong references to in-flight tasks. `asyncio` only holds weak references, so
#: a fire-and-forget task can be garbage-collected mid-run without this.
_tasks: set[asyncio.Task[None]] = set()


def spawn(
    analysis_id: str,
    crop_id: str,
    district_id: str | None,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    question_text: str | None = None,
    audio_bytes: bytes | None = None,
    audio_mime: str | None = None,
) -> None:
    """Start a pipeline run in the background."""
    task = asyncio.create_task(
        run(
            analysis_id,
            crop_id,
            district_id,
            latitude=latitude,
            longitude=longitude,
            question_text=question_text,
            audio_bytes=audio_bytes,
            audio_mime=audio_mime,
        )
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def run(
    analysis_id: str,
    crop_id: str,
    district_id: str | None,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    question_text: str | None = None,
    audio_bytes: bytes | None = None,
    audio_mime: str | None = None,
) -> None:
    """Run the whole pipeline. Never raises: failures are recorded on the row."""
    async with _semaphore_for_current_loop():
        try:
            await _pipeline(
                analysis_id,
                crop_id,
                district_id,
                latitude=latitude,
                longitude=longitude,
                question_text=question_text,
                audio_bytes=audio_bytes,
                audio_mime=audio_mime,
            )
        except Exception as exc:
            log.exception("analysis %s failed", analysis_id)
            await store.mark_failed(analysis_id, exc.__class__.__name__)


def _combine_farmer_question(
    question_text: str | None,
    transcribed_voice: str | None,
) -> str | None:
    parts: list[str] = []
    for item in (question_text, transcribed_voice):
        if item:
            cleaned = " ".join(item.split()).strip()
            if cleaned and cleaned not in parts:
                parts.append(cleaned)
    if not parts:
        return None
    if len(parts) == 1:
        combined = parts[0]
    else:
        first, second = parts[0], parts[1]
        sep = " " if first.endswith(("।", "?", "!", ".", ",")) else "। "
        combined = f"{first}{sep}{second}"
    return combined[: settings.max_question_chars].strip() or None


def _determine_photo_crop(crop_id: str, vision: VisionFacts) -> str:
    inferred_crop = vision.get("inferred_photo_crop")
    if isinstance(inferred_crop, str) and inferred_crop in knowledge_base.SUPPORTED_CROPS:
        return inferred_crop
    vision_crop = vision.get("crop_id")
    if isinstance(vision_crop, str) and vision_crop in knowledge_base.SUPPORTED_CROPS:
        return vision_crop
    label = str(vision.get("label") or "").strip().lower()
    if label.startswith("potato"):
        return "potato"
    if label.startswith("rice"):
        return "rice"
    return crop_id


def _effective_vision_for_diagnosis(vision: VisionFacts, photo_crop: str) -> VisionFacts:
    """Promote a clear cross-crop unmasked detection so diagnosis/RAG match the photo."""
    if (
        vision.get("crop_override_detected")
        and vision.get("override_confirmed")
        and vision.get("override_disease_id")
    ):
        promoted: VisionFacts = {
            **vision,
            "status": "identified",
            "label": vision.get("raw_top_label"),
            "confidence": vision.get("raw_top_confidence"),
            "crop_id": photo_crop,
            "disease_id": vision.get("override_disease_id"),
            "is_healthy": bool(vision.get("override_is_healthy")),
        }
        return promoted
    return vision


def _build_crop_mismatch(
    photo_crop: str,
    selected_crop: str,
    question_ctx: QuestionContext,
    vision: VisionFacts,
) -> dict[str, Any] | None:
    spoken_differs = (
        question_ctx.spoken_crop in ("rice", "potato", "other")
        and question_ctx.spoken_crop != photo_crop
    )
    inferred_crop = vision.get("inferred_photo_crop")
    dropdown_differs = bool(
        isinstance(inferred_crop, str)
        and inferred_crop in knowledge_base.SUPPORTED_CROPS
        and inferred_crop != selected_crop
    )
    if not spoken_differs and not dropdown_differs:
        return None

    photo_crop_bn = _CROP_BN_MAP.get(photo_crop, photo_crop)
    leaf_phrase = (
        "ধানের পাতা"
        if photo_crop == "rice"
        else ("আলুর পাতা" if photo_crop == "potato" else f"{photo_crop_bn} পাতা")
    )
    if spoken_differs:
        question_crop_bn = question_ctx.spoken_crop_bn or _CROP_BN_MAP.get(
            question_ctx.spoken_crop, "অন্য ফসল"
        )
    else:
        question_crop_bn = _CROP_BN_MAP.get(selected_crop, selected_crop)

    if dropdown_differs:
        message_bn = (
            f"ছবিতে {leaf_phrase} দেখা যাচ্ছে, কিন্তু প্রশ্নে/নির্বাচনে {question_crop_bn} "
            "ফসলের কথা বলা হয়েছে। বিভ্রান্তি এড়াতে পরামর্শটি শুধুমাত্র ছবির ফসলের যাচাইকৃত "
            "BARI/BRRI নির্দেশিকায় সীমাবদ্ধ রাখা হয়েছে।"
        )
    else:
        message_bn = (
            f"ছবিতে {leaf_phrase} দেখা যাচ্ছে, কিন্তু আপনার প্রশ্নে {question_crop_bn} "
            "ফসলের কথা বলা হয়েছে। বিভ্রান্তি এড়াতে পরামর্শটি শুধুমাত্র ছবির ফসলের যাচাইকৃত "
            "BARI/BRRI নির্দেশিকায় সীমাবদ্ধ রাখা হয়েছে।"
        )
    return {
        "detected": True,
        "photoCropBn": photo_crop_bn,
        "questionCropBn": question_crop_bn,
        "messageBn": message_bn,
    }


async def _pipeline(
    analysis_id: str,
    crop_id: str,
    district_id: str | None,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    question_text: str | None = None,
    audio_bytes: bytes | None = None,
    audio_mime: str | None = None,
) -> None:
    await store.update_stage(analysis_id, "reading", "processing")

    has_audio = bool(audio_bytes and audio_mime)
    has_text = bool(question_text and question_text.strip())

    if transcribe_bangla_question is not _DEFAULT_TRANSCRIBE:
        if has_audio:
            vision, transcribed_voice = await asyncio.gather(
                _classify(analysis_id, crop_id),
                transcribe_bangla_question(audio_bytes, audio_mime, settings),  # type: ignore[arg-type]
            )
        else:
            vision = await _classify(analysis_id, crop_id)
            transcribed_voice = None
        farmer_question = _combine_farmer_question(question_text, transcribed_voice)
        question_ctx = extract_question_context_offline(
            farmer_question, transcript_bn=transcribed_voice
        )
    elif has_audio or has_text:
        vision, question_ctx = await asyncio.gather(
            _classify(analysis_id, crop_id),
            extract_question_context(audio_bytes, audio_mime, question_text, settings),
        )
        farmer_question = _combine_farmer_question(question_text, question_ctx.transcript_bn)
    else:
        vision = await _classify(analysis_id, crop_id)
        question_ctx = QuestionContext()
        farmer_question = None

    photo_crop = _determine_photo_crop(crop_id, vision)
    effective_vision = _effective_vision_for_diagnosis(vision, photo_crop)
    crop_mismatch = _build_crop_mismatch(photo_crop, crop_id, question_ctx, vision)

    disease, confirmed = _resolve_disease(photo_crop, effective_vision)
    if disease is None:
        await store.mark_failed(analysis_id, "knowledge_base_miss")
        return

    await store.update_stage(analysis_id, "grounding")

    district = get_district(district_id) if district_id else None
    weather = None
    display_district: str | None = None
    resolved_place: str | None = None
    extension_office_bn: str | None = None

    has_voice_location = bool(
        question_ctx.spoken_place_en
        or question_ctx.spoken_place_bn
        or question_ctx.spoken_district_en
    )
    resolved_loc = None
    if has_voice_location:
        resolved_loc = await resolve_precise_location(
            question_ctx.spoken_place_en,
            question_ctx.spoken_place_bn,
            question_ctx.spoken_district_en,
            gps_lat=latitude,
            gps_lon=longitude,
            fallback_district_id=district_id,
        )

    if resolved_loc is not None:
        resolved_place = resolved_loc.resolved_place
        extension_office_bn = build_extension_office_bn(resolved_place)
        weather = await fetch_weather(
            latitude=resolved_loc.latitude,
            longitude=resolved_loc.longitude,
            district_id=resolved_loc.district_id,
            district_name=resolved_loc.district_name,
            location_source="voice",
            resolved_place=resolved_place,
            location_candidates=[c.as_contract_dict() for c in resolved_loc.candidates],
            extension_office_bn=extension_office_bn,
        )
        display_district = resolved_loc.district_name
    elif latitude is not None and longitude is not None:
        nearest = find_nearest_district(latitude, longitude)
        weather = await fetch_weather(
            latitude=latitude,
            longitude=longitude,
            district_id=str(nearest["id"]),
            district_name=str(nearest["name"]),
            location_source="gps",
        )
        display_district = str(district["name"]) if district else str(nearest["name"])
    elif district is not None:
        weather = await fetch_weather(
            district_id=str(district["id"]),
            district_name=str(district["name"]),
            location_source="district",
        )
        display_district = str(district["name"])

    if weather is not None:
        if weather.resolved_place and not resolved_place:
            resolved_place = weather.resolved_place
        if weather.extension_office_bn and not extension_office_bn:
            extension_office_bn = weather.extension_office_bn

    rag_disease_id = disease["id"] if confirmed else "uncertain"
    if confirmed:
        query_parts = [disease["diseaseLabel"], disease["bnName"], *disease["symptoms"][:2]]
    else:
        query_parts = [disease["cropName"], "scouting", "SAAO", "clear photo"]
    if weather is not None and not weather.spray_window_ok:
        query_parts.append("rain wind spray warning")

    # Hard RAG guard: on crop mismatch, never let the mismatched spoken crop
    # influence retrieval; stick strictly to the photographed crop & diagnosis.
    if crop_mismatch is not None:
        rag_query = " ".join(query_parts)
    else:
        rag_query = farmer_question if farmer_question else " ".join(query_parts)

    grounding = await asyncio.to_thread(
        rag.retrieve_grounding_bundle,
        rag_disease_id,
        photo_crop,
        rag_query,
        3,
    )

    facts = _build_facts(
        disease,
        weather,
        effective_vision,
        confirmed=confirmed,
        grounding=grounding,
        farmer_question=farmer_question,
        resolved_place=resolved_place,
        spoken_plot_hint_bn=question_ctx.spoken_plot_hint_bn,
        extension_office_bn=extension_office_bn,
        crop_mismatch=crop_mismatch,
    )

    await store.update_stage(analysis_id, "writing")
    narrative, generated_by = await _write(facts)

    result = _build_result(
        analysis_id,
        disease,
        weather,
        narrative,
        generated_by,
        vision=effective_vision,
        confirmed=confirmed,
        grounding=grounding,
        farmer_question=farmer_question,
        district_name=display_district,
        crop_mismatch=crop_mismatch,
    )
    await store.save_result(analysis_id, result, disease_id=disease["id"])

    # Text is complete and served now. Speech continues in the background, so
    # nobody waits for a voice while reading advice that is already there -
    # `/audio` answers 202 until the file exists.
    if settings.tts_enabled:
        await store.update_stage(analysis_id, "speaking", "completed")
        task = asyncio.create_task(_speak(analysis_id, narrative, disease, grounding))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    else:
        await store.update_stage(analysis_id, "done", "completed")


async def _classify(analysis_id: str, crop_id: str) -> VisionFacts:
    """Run the classifier, guarding only against a missing image.

    The classifier itself never raises; this exists so that a deleted file after
    upload reads as "no model output" rather than a failed job.
    """
    image_path = settings.images_dir / f"{analysis_id}.jpg"
    if not image_path.exists():
        log.warning("analysis %s has no stored image; skipping vision", analysis_id)
        return VisionFacts(
            status="unavailable",
            label=None,
            confidence=None,
            crop_id=None,
            disease_id=None,
            is_healthy=False,
            model=None,
        )
    return await get_classifier().classify(image_path, crop_id)


def _resolve_disease(
    crop_id: str, vision: VisionFacts
) -> tuple[knowledge_base.DiseaseEntry | None, bool]:
    """Pick the KB entry the advice will be built on and whether it was confirmed.

    A classifier match wins — including a `healthy` match, which resolves to the
    crop's healthy entry and carries no chemical. Everything else falls back to
    one curated entry per crop and is marked unconfirmed downstream.
    """
    if vision["status"] == "identified" and vision["disease_id"]:
        matched = knowledge_base.get_disease(vision["disease_id"])
        if matched is not None:
            return matched, True

    fallback_id = FALLBACK_DISEASE.get(crop_id)
    return (knowledge_base.get_disease(fallback_id) if fallback_id else None), False


def _confidence_band(confidence: float | None) -> str:
    if confidence is None:
        return "low"
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    if confidence < LOW_CONFIDENCE:
        return "low"
    return "medium"


async def _write(facts: dict[str, Any]) -> tuple[str, str]:
    writer = get_writer()
    try:
        return await writer.write(facts)
    except (InvalidNarrative, Exception) as exc:  # noqa: BLE001
        log.warning("LLM write failed (%s); using template", exc.__class__.__name__)
        from ..llm.writer import TemplateWriter

        return await TemplateWriter().write(facts)


async def _speak(
    analysis_id: str,
    narrative: str,
    disease: knowledge_base.DiseaseEntry,
    grounding: rag.GroundingBundle,
) -> None:
    """Synthesise in the background, then close out the stage.

    Every exit path moves the stage to `done`. Leaving a row at `speaking` after
    a TTS failure would mean a result that never looks finished.
    """
    try:
        await _speak_inner(analysis_id, narrative, disease, grounding)
    finally:
        await store.update_stage(analysis_id, "done", "completed")


async def _speak_inner(
    analysis_id: str,
    narrative: str,
    disease: knowledge_base.DiseaseEntry,
    grounding: rag.GroundingBundle,
) -> None:
    engine = get_engine()
    if not engine.available:
        log.warning(
            "TTS disabled/unavailable for %s (tts_enabled=%s)",
            analysis_id,
            settings.tts_enabled,
        )
        await store.update_audio(analysis_id, {"status": "failed", "url": None})
        return

    await store.update_audio(analysis_id, {"status": "pending", "url": None})
    terms = list(
        dict.fromkeys(
            [chemical["bnName"] for chemical in disease["chemical"]]
            + [chemical["bn_name"] for chemical in grounding.chemicals]
        )
    )
    try:
        clip = await engine.synthesise(narrative, settings.audio_dir, required_terms=terms)
    except TtsSafetyError as exc:
        log.error("TTS refused to synthesise for %s: %s", analysis_id, exc)
        await store.update_audio(analysis_id, {"status": "failed", "url": None})
        return
    except Exception as exc:
        log.warning(
            "TTS failed for %s: %s: %s",
            analysis_id,
            exc.__class__.__name__,
            str(exc) or "no detail",
            exc_info=True,
        )
        await store.update_audio(analysis_id, {"status": "failed", "url": None})
        return

    await store.update_audio(
        analysis_id,
        {
            "status": "ready",
            "url": f"/api/v1/analyses/{analysis_id}/audio",
            "durationMs": clip.duration_ms,
            "byteSize": clip.byte_size,
            "file": clip.path.name,
        },
    )


def _build_facts(
    disease: knowledge_base.DiseaseEntry,
    weather: Any,
    vision: VisionFacts,
    *,
    confirmed: bool,
    grounding: rag.GroundingBundle,
    farmer_question: str | None = None,
    resolved_place: str | None = None,
    spoken_plot_hint_bn: str | None = None,
    extension_office_bn: str | None = None,
    crop_mismatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The exact shape handed to the LLM. Facts only, no instructions.

    `vision` is what the classifier actually returned. When it did not identify
    anything, the facts say so and carry no confidence, so the model cannot
    write "symptoms were observed at medium confidence" about a photo it never
    saw a result for.
    """
    is_healthy = bool(vision.get("is_healthy")) or disease["id"].endswith("_healthy")
    cultural_only = (
        grounding.cultural_only
        if grounding.sources
        else (is_healthy or not disease["chemical"])
    )
    allow_chemicals = confirmed and not cultural_only and not is_healthy
    mismatch_detected = bool(crop_mismatch and crop_mismatch.get("detected"))
    refer_to_officer = (not confirmed) or grounding.refer_to_officer or mismatch_detected

    if not allow_chemicals:
        prompt_chemicals: list[dict[str, str]] = []
    elif grounding.chemicals:
        prompt_chemicals = [
            {
                "product": chemical["active_ingredient"],
                "bnName": chemical["bn_name"],
                "dose": chemical["dose"],
                "phi": chemical["phi"],
            }
            for chemical in grounding.chemicals
        ]
    else:
        prompt_chemicals = [
            {
                "product": chemical["product"],
                "bnName": chemical["bnName"],
                "dose": chemical["dose"],
                "note": chemical["note"],
            }
            for chemical in disease["chemical"]
        ]

    # Exclude chemical-bearing RAG prose whenever chemicals are disallowed so the
    # writer never sees fungicide names (e.g. Mancozeb) that the validator blocks.
    passages = [
        {
            **passage,
            "chemicals": passage["chemicals"] if allow_chemicals else [],
        }
        for passage in grounding.prompt_passages
        if allow_chemicals or not passage.get("chemicals")
    ]

    weather_facts = weather.as_prompt_facts() if weather else None
    if weather_facts is not None and not allow_chemicals:
        weather_facts["safest_window_summary_bn"] = None

    pure_extension_bn = (
        extension_office_bn.replace("(SAAO)", "").strip() if extension_office_bn else None
    )
    facts_mismatch = (
        {
            "detected": True,
            "photo_crop_bn": crop_mismatch["photoCropBn"],
            "question_crop_bn": crop_mismatch["questionCropBn"],
            "selection_mismatch": "নির্বাচনে" in str(crop_mismatch.get("messageBn", "")),
            "note_bn": (
                f"ছবিতে {crop_mismatch['photoCropBn']} পাতা দেখা যাচ্ছে, "
                f"কিন্তু প্রশ্ন বা নির্বাচনে {crop_mismatch['questionCropBn']} ফসলের কথা বলা হয়েছে।"
            ),
        }
        if mismatch_detected and crop_mismatch is not None
        else None
    )

    return {
        "diagnosis": {
            "crop": disease["cropName"],
            # The short name, so the narrative reads "ধানে ব্লাস্ট" rather than
            # "ধানে ধানের ব্লাস্ট". The full label is used in the result card.
            "disease": f"{disease['bnName']} রোগ" if not is_healthy else disease["bnName"],
            "confidence": _confidence_label(vision["confidence"]) if confirmed else None,
            "confirmed_by": "ছবি বিশ্লেষণ" if confirmed else None,
            "healthy": is_healthy,
        },
        # A healthy plant has no symptoms to describe; being explicit stops
        # any future edit from making "no disease" read like a symptom.
        "symptoms_observed": disease["symptoms"] if (confirmed and not is_healthy) else [],
        "treatment": {
            "organic": disease["organic"],
            "chemical": prompt_chemicals,
        },
        "prevention": disease["prevention"],
        "weather": weather_facts,
        "uncertain": not confirmed,
        "farmer_question": farmer_question,
        "resolved_place": resolved_place,
        "spoken_plot_hint_bn": spoken_plot_hint_bn,
        "extension_office_bn": pure_extension_bn,
        "crop_mismatch": facts_mismatch,
        "rag_grounding": {
            "disease_id": grounding.normalized_disease_id or disease["id"],
            "cultural_only": cultural_only,
            "refer_to_officer": refer_to_officer,
            "is_healthy": is_healthy,
            "ppe_reminder": grounding.ppe_reminder_bn if allow_chemicals else None,
            "passages": passages,
        },
    }


def _confidence_label(confidence: float | None) -> str:
    """Bangla band name, matching what the UI shows."""
    return {"high": "বেশি", "medium": "মাঝারি", "low": "কম"}[_confidence_band(confidence)]


def _treatment_steps(
    disease: knowledge_base.DiseaseEntry,
    grounding: rag.GroundingBundle,
    *,
    confirmed: bool,
    cultural_only: bool,
) -> list[str]:
    if not confirmed or cultural_only:
        return list(disease["organic"])

    if grounding.chemicals:
        chem_steps = [
            f"{chem['bn_name']}: {chem['dose']} (ফসল তোলার আগে বিরতি: {chem['phi']})"
            for chem in grounding.chemicals
        ]
    else:
        chem_steps = [
            f"{chemical['bnName']}: {chemical['dose']} ({chemical['note']})"
            for chemical in disease["chemical"]
        ]
    return chem_steps + list(disease["organic"])


def _build_result(
    analysis_id: str,
    disease: knowledge_base.DiseaseEntry,
    weather: Any,
    narrative: str,
    generated_by: str,
    *,
    vision: VisionFacts,
    confirmed: bool,
    grounding: rag.GroundingBundle,
    farmer_question: str | None = None,
    district_name: str | None = None,
    crop_mismatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    confidence = vision["confidence"]
    is_healthy = bool(vision.get("is_healthy")) or disease["id"].endswith("_healthy")
    cultural_only = (
        grounding.cultural_only
        if grounding.sources
        else (is_healthy or not disease["chemical"])
    )
    allow_chemicals = confirmed and not cultural_only and not is_healthy
    mismatch_detected = bool(crop_mismatch and crop_mismatch.get("detected"))
    refer_to_officer = (not confirmed) or grounding.refer_to_officer or mismatch_detected
    ppe_reminder = (
        grounding.ppe_reminder_bn if (allow_chemicals and grounding.ppe_reminder_bn) else None
    )
    chemical_chunk_ids = {
        chunk["chunk_id"] for chunk in grounding.chunks if chunk.get("chemicals")
    }
    visible_sources = [
        source
        for source in grounding.sources
        if allow_chemicals or source.chunk_id not in chemical_chunk_ids
    ]

    if confirmed:
        if is_healthy:
            summary = "গাছ সুস্থ আছে, কোনো রাসায়নিক স্প্রে করার প্রয়োজন নেই"
            uncertainty_message = (
                "আপনার গাছটি সুস্থ দেখাচ্ছে। নিয়মিত পরিচর্যা চালিয়ে যান এবং "
                "রোগের লক্ষণ দেখা দিলে আবার ছবি দিন।"
            )
        elif cultural_only:
            summary = "এটি ছত্রাকজনিত রোগ নয়; ছত্রাকনাশক স্প্রে কাজ করবে না। নিচের পরিচর্যাগুলো অনুসরণ করুন"
            uncertainty_message = "এটি একটি আনুমানিক ফলাফল। নিশ্চিত হওয়ার জন্য কৃষি কর্মকর্তার পরামর্শ নিন।"
        else:
            summary = "নিচের ধাপগুলো অনুসরণ করুন"
            uncertainty_message = "এটি একটি আনুমানিক ফলাফল। নিশ্চিত হওয়ার জন্য কৃষি কর্মকর্তার পরামর্শ নিন।"
    else:
        summary = f"{disease['cropName']} ফসলের জন্য সাধারণ পরিচর্যা"
        uncertainty_message = (
            "ছবি থেকে রোগ নিশ্চিতভাবে শনাক্ত করা যায়নি। নিচের পরামর্শটি "
            f"{disease['cropName']} ফসলের সাধারণ নির্দেশনা। সঠিক রোগ নিশ্চিত করতে "
            "আরও পরিষ্কার ছবি দিন, অথবা উপজেলা কৃষি কর্মকর্তার পরামর্শ নিন।"
        )

    return {
        "id": analysis_id,
        "createdAt": datetime.now(UTC).isoformat(),
        "imageUrl": f"/api/v1/analyses/{analysis_id}/image",
        "cropName": disease["cropName"],
        "status": "completed",
        "stage": "speaking" if settings.tts_enabled else "done",
        "diagnosis": {
            "diseaseLabel": disease["diseaseLabel"],
            "cropName": disease["cropName"],
            # Still null on a miss: the label shown is the fallback entry, and
            # attaching a number to it would be inventing evidence.
            "confidence": round(confidence, 3) if (confirmed and confidence is not None) else None,
            "confidenceBand": (
                _confidence_band(confidence) if (confirmed and confidence is not None) else "low"
            ),
            "isAiEstimate": True,
        },
        "symptoms": disease["symptoms"] if (confirmed and not is_healthy) else [],
        "treatment": {
            "summary": summary,
            "steps": _treatment_steps(
                disease,
                grounding,
                confirmed=confirmed,
                cultural_only=cultural_only or is_healthy,
            ),
            "source": visible_sources[0].source_title if visible_sources else disease["source"],
        },
        "prevention": disease["prevention"],
        "needsProfessionalVerification": True,
        "ragSources": [source.to_contract_dict() for source in visible_sources],
        "ppeReminder": ppe_reminder,
        "referToOfficer": refer_to_officer,
        "uncertainty": {
            "isUncertain": not confirmed,
            "message": uncertainty_message,
        },
        "narrative": narrative,
        "farmerQuestion": farmer_question,
        "generatedBy": generated_by,
        "weather": weather.as_contract_dict() if weather else None,
        "cropMismatch": crop_mismatch,
        "audio": {"status": "pending" if settings.tts_enabled else "none", "url": None},
        "district": district_name,
    }
