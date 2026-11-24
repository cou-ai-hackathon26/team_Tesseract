"""Tests for the 8-class BARI/BRRI RAG corpus, retriever, and grounded pipeline."""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import API_ROOT, settings
from app.contracts import AnalysisResult
from app.grounding import knowledge_base, rag
from app.jobs import pipeline
from app.llm.writer import GeminiWriter, InvalidNarrative, TemplateWriter, reset_writer
from app.main import app
from app.vision import VisionFacts
from app.vision.models import EFFICIENTNET

CORPUS_PATH = API_ROOT / "data" / "bari_brri_corpus.json"
INDEX_PATH = API_ROOT / "data" / "bari_brri_index.json"

REQUIRED_CHUNK_KEYS = {
    "chunk_id",
    "disease_id",
    "aliases",
    "crop_id",
    "source_title",
    "section",
    "title_bn",
    "content_en",
    "content_bn",
    "cultural_only",
    "chemicals",
    "ppe_reminder_bn",
    "refer_to_officer",
}

_LATIN_RE = re.compile(r"[A-Za-z]")


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (40, 120, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _wait(client: TestClient, analysis_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summary = client.get(f"/api/v1/analyses/{analysis_id}").json()
        if summary["status"] in {"completed", "failed"}:
            return summary
        time.sleep(0.1)
    raise AssertionError("analysis did not finish")


def test_corpus_covers_all_8_classes_with_required_schema() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert isinstance(corpus, list)
    assert len(corpus) >= 8

    disease_ids = {chunk["disease_id"] for chunk in corpus}
    assert disease_ids == set(EFFICIENTNET["labels"])

    for chunk in corpus:
        assert set(chunk.keys()) >= REQUIRED_CHUNK_KEYS, chunk.get("chunk_id")
        assert chunk["content_en"].strip()
        assert chunk["content_bn"].strip()
        assert isinstance(chunk["cultural_only"], bool)
        assert isinstance(chunk["refer_to_officer"], bool)
        assert isinstance(chunk["aliases"], list)
        assert isinstance(chunk["chemicals"], list)
        for bn_field in ("title_bn", "content_bn", "ppe_reminder_bn"):
            assert not _LATIN_RE.search(chunk[bn_field]), (
                f"{chunk['chunk_id']} leaked Latin in {bn_field}"
            )
        for chem in chunk["chemicals"]:
            assert set(chem.keys()) >= {"active_ingredient", "bn_name", "dose", "phi"}
            assert not _LATIN_RE.search(chem["bn_name"])
            assert not _LATIN_RE.search(chem["dose"])
            assert not _LATIN_RE.search(chem["phi"])


def test_precomputed_index_matches_corpus_chunks() -> None:
    corpus = rag.load_corpus(CORPUS_PATH)
    model_name, dims, vectors = rag.load_index(INDEX_PATH)
    assert model_name
    assert dims > 0
    assert set(vectors.keys()) == {chunk["chunk_id"] for chunk in corpus}


@pytest.mark.parametrize(
    ("disease_id", "crop_id", "expected_cultural", "expected_refer"),
    [
        ("potato_early_blight", "potato", False, False),
        ("potato_late_blight", "potato", False, False),
        ("potato_healthy", "potato", True, False),
        ("rice_blast", "rice", False, False),
        ("rice_brown_spot", "rice", False, False),
        ("rice_bacterial_leaf_blight", "rice", True, True),
        ("rice_tungro", "rice", True, True),
        ("rice_healthy", "rice", True, False),
    ],
)
def test_retrieve_grounding_for_all_8_classes(
    monkeypatch: pytest.MonkeyPatch,
    disease_id: str,
    crop_id: str,
    expected_cultural: bool,
    expected_refer: bool,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    rag.reset_rag_cache()

    sources = rag.retrieve_grounding(disease_id, crop_id)
    assert len(sources) >= 1
    assert all(src.disease_id == disease_id for src in sources)
    assert all("BARI" in src.source_title or "BRRI" in src.source_title for src in sources)

    bundle = rag.retrieve_grounding_bundle(disease_id, crop_id)
    assert bundle.cultural_only is expected_cultural
    assert bundle.refer_to_officer is expected_refer
    if expected_cultural:
        assert bundle.chemicals == []
        assert bundle.ppe_reminder_bn is None
    else:
        assert len(bundle.chemicals) >= 1
        assert all(chem["phi"] for chem in bundle.chemicals)
        assert bundle.ppe_reminder_bn


def test_blb_and_tungro_explicitly_forbid_fungicides() -> None:
    blb = rag.retrieve_grounding_bundle("rice_bacterial_leaf_blight", "rice")
    assert blb.cultural_only is True
    assert blb.refer_to_officer is True
    assert blb.chemicals == []
    blb_en = " ".join(c["content_en"] for c in blb.chunks).lower()
    assert "bacterial" in blb_en
    assert "fungicides do not" in blb_en
    assert "potash" in blb_en or "mop" in blb_en
    assert "saao" in blb_en

    tungro = rag.retrieve_grounding_bundle("rice_tungro", "rice")
    assert tungro.cultural_only is True
    assert tungro.refer_to_officer is True
    assert tungro.chemicals == []
    tungro_en = " ".join(c["content_en"] for c in tungro.chunks).lower()
    assert "viral" in tungro_en
    assert "green leafhopper" in tungro_en
    assert "fungicides do not" in tungro_en
    assert "saao" in tungro_en


def test_healthy_safety_protocols() -> None:
    for healthy_id, crop in (("potato_healthy", "potato"), ("rice_healthy", "rice")):
        bundle = rag.retrieve_grounding_bundle(healthy_id, crop)
        assert bundle.cultural_only is True
        assert bundle.refer_to_officer is False
        assert bundle.chemicals == []
        assert "no fungicide" in " ".join(c["content_en"] for c in bundle.chunks).lower()


def test_aliases_resolve_in_rag_and_knowledge_base() -> None:
    assert rag.normalize_disease_id("rice_bacterial_blight") == "rice_bacterial_leaf_blight"
    sources = rag.retrieve_grounding("rice_bacterial_blight", "rice")
    assert sources
    assert all(src.disease_id == "rice_bacterial_leaf_blight" for src in sources)

    blb_from_alias = knowledge_base.get_disease("rice_bacterial_leaf_blight")
    blb_from_legacy = knowledge_base.get_disease("rice_bacterial_blight")
    assert blb_from_alias is not None
    assert blb_from_alias is blb_from_legacy
    assert blb_from_alias["chemical"] == []

    assert knowledge_base.get_disease("rice_tungro") is not None
    assert knowledge_base.get_disease("rice_brown_spot") is not None
    assert knowledge_base.get_disease("rice_healthy") is not None
    assert knowledge_base.get_disease("potato_healthy") is not None


def test_hybrid_tfidf_and_vector_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    rag.reset_rag_cache()

    # Offline TF-IDF: querying fungicide/dose ranks chunk 02 ahead of chunk 01.
    chem_first = rag.retrieve_grounding(
        "rice_blast",
        "rice",
        query_text="Tricyclazole Isoprothiolane fungicide dose",
        top_k=1,
    )
    assert chem_first[0].chunk_id == "brri_rice_blast_02"

    # Simulated Gemini vector similarity using an exact stored chunk vector.
    _, _, vectors = rag.load_index(INDEX_PATH)
    target_vec = vectors["brri_rice_blast_01"]
    monkeypatch.setattr(rag, "_embed_query", lambda *_args: target_vec)

    vec_first = rag.retrieve_grounding(
        "rice_blast",
        "rice",
        query_text="খুব হালকা দাগ",
        top_k=1,
    )
    assert vec_first[0].chunk_id == "brri_rice_blast_01"


async def test_gemini_writer_receives_rag_passages_and_blocks_cultural_only_chemicals() -> None:
    bundle = rag.retrieve_grounding_bundle("rice_bacterial_leaf_blight", "rice")
    disease = knowledge_base.get_disease("rice_bacterial_leaf_blight")
    assert disease is not None

    facts = pipeline._build_facts(
        disease,
        weather=None,
        vision=VisionFacts(
            status="identified",
            label="rice_bacterial_leaf_blight",
            confidence=0.88,
            crop_id="rice",
            disease_id="rice_bacterial_blight",
            is_healthy=False,
            model="efficientnet",
        ),
        confirmed=True,
        grounding=bundle,
    )
    assert facts["treatment"]["chemical"] == []
    assert facts["rag_grounding"]["cultural_only"] is True
    assert len(facts["rag_grounding"]["passages"]) == 2

    hallucinated_fungicide = (
        "আপনার ধান গাছে ব্যাকটেরিয়াল ব্লাইট হয়েছে। "
        "দ্রুত ট্রাইসাইক্লাজল স্প্রে করুন এবং পানি জমিয়ে রাখুন। "
        "স্থানীয় উপ-সহকারী কৃষি কর্মকর্তার পরামর্শ নিন। এবং জমিতে পর্যাপ্ত সার ও পানি প্রয়োগ নিশ্চিত করুন।"
    )
    valid_cultural_advice = (
        "আপনার ধান গাছে ব্যাকটেরিয়াল লিফ ব্লাইট হয়েছে, তাই কোনো ছত্রাকনাশক কাজ করবে না। "
        "জমির পানি সরিয়ে শুকিয়ে নিন, ইউরিয়া বন্ধ করে পটাশ দিন এবং উপ-সহকারী কৃষি কর্মকর্তার পরামর্শ নিন।"
    )

    seen_prompts: list[str] = []

    class StubModels:
        def __init__(self, payloads: list[str]) -> None:
            self._payloads = list(payloads)

        def generate_content(self, **kwargs):
            seen_prompts.append(kwargs["contents"])
            spoken = json.dumps({"spoken": self._payloads.pop(0)}, ensure_ascii=False)
            return SimpleNamespace(text=spoken)

    writer = GeminiWriter.__new__(GeminiWriter)
    writer._semaphore = asyncio.Semaphore(3)
    writer._client = SimpleNamespace(
        models=StubModels([hallucinated_fungicide, valid_cultural_advice])
    )

    text, generated_by = await writer.write(facts)
    assert generated_by == "llm"
    assert "ছত্রাকনাশক কাজ করবে না" in text
    assert len(seen_prompts) == 2
    assert "BRRI" in seen_prompts[0]

    # Repeated hallucination raises InvalidNarrative so pipeline degrades safely.
    bad_writer = GeminiWriter.__new__(GeminiWriter)
    bad_writer._semaphore = asyncio.Semaphore(3)
    bad_writer._client = SimpleNamespace(
        models=StubModels([hallucinated_fungicide, hallucinated_fungicide])
    )
    with pytest.raises(InvalidNarrative, match="unusable_after_retry"):
        await bad_writer.write(facts)

    # TemplateWriter also handles cultural_only cleanly without recommending chemicals.
    tpl_text, _ = await TemplateWriter().write(facts)
    assert "ছত্রাকনাশক স্প্রে কাজ করবে না" in tpl_text
    assert "উপ-সহকারী কৃষি কর্মকর্তা" in tpl_text


@pytest.mark.parametrize(
    ("label", "crop", "status", "disease_id", "expect_uncertain", "expect_refer", "expect_ppe"),
    [
        ("rice_blast", "rice", "identified", "rice_blast", False, False, True),
        (
            "rice_bacterial_leaf_blight",
            "rice",
            "identified",
            "rice_bacterial_blight",
            False,
            True,
            False,
        ),
        ("rice_tungro", "rice", "identified", "rice_tungro", False, True, False),
        ("potato_healthy", "potato", "identified", "potato_healthy", False, False, False),
    ],
)
def test_pipeline_surfaces_rag_fields_in_api_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    crop: str,
    status: str,
    disease_id: str | None,
    expect_uncertain: bool,
    expect_refer: bool,
    expect_ppe: bool,
) -> None:
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    reset_writer()
    rag.reset_rag_cache()

    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    async def fake_classify(self, image_path, crop_id):
        return VisionFacts(
            status=status,  # type: ignore[arg-type]
            label=label,
            confidence=0.92,
            crop_id=crop,
            disease_id=disease_id,
            is_healthy=label.endswith("_healthy"),
            model="efficientnet",
        )

    monkeypatch.setattr(
        pipeline.get_classifier().__class__, "classify", fake_classify, raising=False
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/analyses",
            files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
            data={"crop": crop},
        )
        assert created.status_code == 202
        analysis_id = created.json()["id"]
        _wait(client, analysis_id)

        payload = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
        result = AnalysisResult.model_validate(payload)
        assert len(result.ragSources) >= 1
        assert result.uncertainty.isUncertain is expect_uncertain
        assert result.referToOfficer is expect_refer
        if expect_ppe:
            assert result.ppeReminder
        else:
            assert result.ppeReminder is None


def test_unconfirmed_potato_hides_chemical_rag_prose_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    rag.reset_rag_cache()

    disease = knowledge_base.get_disease("potato_late_blight")
    assert disease is not None

    question = "আমার আলুর পাতায় কালো দাগ পড়ছে, এখন কী ওষুধ দেব?"
    bundle = rag.retrieve_grounding_bundle("potato_late_blight", "potato", question, top_k=3)
    assert any(chunk.get("chemicals") for chunk in bundle.chunks)

    vision = VisionFacts(
        status="undetermined",
        label=None,
        confidence=None,
        crop_id=None,
        disease_id=None,
        is_healthy=False,
        model=None,
    )
    facts = pipeline._build_facts(
        disease,
        weather=None,
        vision=vision,
        confirmed=False,
        grounding=bundle,
        farmer_question=question,
    )
    assert facts["treatment"]["chemical"] == []
    assert [p["chunk_id"] for p in facts["rag_grounding"]["passages"]] == [
        "bari_potato_late_blight_01"
    ]
    assert all("ম্যানকোজেব" not in p["content_bn"] for p in facts["rag_grounding"]["passages"])

    result = pipeline._build_result(
        "test-id",
        disease,
        weather=None,
        narrative="আপনার আলু গাছের ছবি শনাক্ত করা যায়নি।",
        generated_by="template",
        vision=vision,
        confirmed=False,
        grounding=bundle,
        farmer_question=question,
    )
    assert [s["chunkId"] for s in result["ragSources"]] == ["bari_potato_late_blight_01"]


def test_uncertain_rag_fallback_for_rice_potato_and_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    rag.reset_rag_cache()

    rice_bundle = rag.retrieve_grounding_bundle("uncertain", "rice", "ধানের পাতায় দাগ", top_k=3)
    assert rice_bundle.normalized_disease_id == "uncertain"
    assert rice_bundle.cultural_only is True
    assert rice_bundle.refer_to_officer is True
    assert rice_bundle.chemicals == []
    assert [s.chunk_id for s in rice_bundle.sources] == ["brri_rice_healthy_02"]

    for query in (None, "আমার আলুর পাতায় দাগ, কোন ছত্রাকনাশক স্প্রে করব?"):
        potato_bundle = rag.retrieve_grounding_bundle("uncertain", "potato", query, top_k=3)
        assert potato_bundle.normalized_disease_id == "uncertain"
        assert potato_bundle.cultural_only is True
        assert potato_bundle.refer_to_officer is True
        assert potato_bundle.chemicals == []
        chunk_ids = [s.chunk_id for s in potato_bundle.sources]
        assert chunk_ids == ["bari_potato_healthy_02"]
        assert all("blight" not in s.disease_id for s in potato_bundle.sources)
        assert "bari_potato_healthy_01" not in chunk_ids

    unknown_bundle = rag.retrieve_grounding_bundle("uncertain", "unknown", None, top_k=3)
    assert unknown_bundle.normalized_disease_id == "uncertain"
    assert unknown_bundle.cultural_only is True
    assert unknown_bundle.refer_to_officer is True
    assert unknown_bundle.chemicals == []
    assert [s.chunk_id for s in unknown_bundle.sources] == [
        "brri_rice_healthy_02",
        "bari_potato_healthy_02",
    ]


def test_pipeline_unconfirmed_rice_uses_safe_scouting_rag_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "vision_enabled", False)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    reset_writer()
    rag.reset_rag_cache()

    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/analyses",
            files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
            data={"crop": "rice"},
        )
        assert created.status_code == 202
        analysis_id = created.json()["id"]
        _wait(client, analysis_id)

        payload = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
        result = AnalysisResult.model_validate(payload)
        assert result.uncertainty.isUncertain is True
        assert result.referToOfficer is True
        assert result.ppeReminder is None
        assert [src.chunkId for src in result.ragSources] == ["brri_rice_healthy_02"]
