"""Lightweight hybrid BARI/BRRI RAG retriever.

Combines deterministic disease/crop filtering with cosine similarity over
precomputed Gemini embeddings (`apps/api/data/bari_brri_index.json`) and a
zero-network TF-IDF fallback whenever offline or unconfigured.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from pydantic import BaseModel

from ..config import API_ROOT, settings

log = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = API_ROOT / "data" / "bari_brri_corpus.json"
DEFAULT_INDEX_PATH = API_ROOT / "data" / "bari_brri_index.json"

_TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[a-z0-9_]+")

_VISION_ALIAS_MAP: dict[str, str] = {
    "rice_bacterial_blight": "rice_bacterial_leaf_blight",
    "rice_bacterial_leaf_blight": "rice_bacterial_leaf_blight",
}

_CROP_ALIAS_MAP: dict[str, str] = {
    "rice": "rice",
    "ধান": "rice",
    "potato": "potato",
    "আলু": "potato",
    "tomato": "tomato",
    "টমেটো": "tomato",
    "chili": "chili",
    "মরিচ": "chili",
}


class RagChemical(TypedDict):
    active_ingredient: str
    bn_name: str
    dose: str
    phi: str


class RagChunk(TypedDict):
    chunk_id: str
    disease_id: str
    aliases: list[str]
    crop_id: str
    source_title: str
    section: str
    title_bn: str
    content_en: str
    content_bn: str
    cultural_only: bool
    chemicals: list[RagChemical]
    ppe_reminder_bn: str
    refer_to_officer: bool


class RagSource(BaseModel):
    """Structured citation returned to callers and surfaced in the API contract."""

    chunk_id: str
    source_title: str
    section: str
    disease_id: str
    excerpt: str
    cultural_only: bool
    refer_to_officer: bool

    def to_contract_dict(self) -> dict[str, Any]:
        return {
            "chunkId": self.chunk_id,
            "sourceTitle": self.source_title,
            "section": self.section,
            "diseaseId": self.disease_id,
            "excerpt": self.excerpt,
            "culturalOnly": self.cultural_only,
            "referToOfficer": self.refer_to_officer,
        }


@dataclass(frozen=True, slots=True)
class GroundingBundle:
    """Retrieved BARI/BRRI context consumed by the pipeline and writer."""

    normalized_disease_id: str
    sources: list[RagSource]
    chunks: list[RagChunk]
    cultural_only: bool
    refer_to_officer: bool
    chemicals: list[RagChemical]
    ppe_reminder_bn: str | None
    prompt_passages: list[dict[str, Any]]


def chunk_search_text(chunk: RagChunk) -> str:
    """Build searchable text used both when indexing and for offline TF-IDF."""
    chem_parts = [
        f"{chem['active_ingredient']} {chem['bn_name']} {chem['dose']} {chem['phi']}"
        for chem in chunk.get("chemicals", [])
    ]
    parts = [
        chunk["disease_id"],
        chunk["crop_id"],
        " ".join(chunk.get("aliases", [])),
        chunk["source_title"],
        chunk["section"],
        chunk["title_bn"],
        chunk["content_en"],
        chunk["content_bn"],
        " ".join(chem_parts),
    ]
    return " ".join(part for part in parts if part)


@lru_cache(maxsize=4)
def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> list[RagChunk]:
    """Load the curated BARI/BRRI JSON corpus from disk."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in RAG corpus {path}")
    return raw


@lru_cache(maxsize=4)
def load_index(path: Path = DEFAULT_INDEX_PATH) -> tuple[str, int, dict[str, np.ndarray]]:
    """Load precomputed unit-normalized chunk embeddings if present."""
    if not path.exists():
        return "", 0, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to read RAG index %s: %s", path, exc)
        return "", 0, {}

    model_name = str(payload.get("model") or "")
    dimensions = int(payload.get("dimensions") or 0)
    raw_vectors = payload.get("embeddings") or {}
    if not isinstance(raw_vectors, dict):
        return "", 0, {}

    vectors: dict[str, np.ndarray] = {}
    for chunk_id, values in raw_vectors.items():
        if not isinstance(values, list) or not values:
            continue
        vec = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vectors[str(chunk_id)] = vec / norm

    return model_name, dimensions, vectors


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    mapping = dict(_VISION_ALIAS_MAP)
    for chunk in load_corpus():
        disease_id = chunk["disease_id"]
        mapping[disease_id.lower()] = disease_id
        for alias in chunk.get("aliases", []):
            key = alias.strip().lower()
            if key and key not in mapping:
                mapping[key] = disease_id
    return mapping


def normalize_disease_id(disease_id: str) -> str:
    """Map checkpoint labels and legacy aliases onto canonical corpus disease IDs."""
    cleaned = disease_id.strip().lower()
    if not cleaned:
        return ""
    return _alias_index().get(cleaned, cleaned)


def _normalize_crop_id(crop_id: str) -> str:
    cleaned = crop_id.strip().lower()
    return _CROP_ALIAS_MAP.get(cleaned, cleaned)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _tfidf_index() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    corpus = load_corpus()
    n_docs = len(corpus)
    if n_docs == 0:
        return {}, {}

    doc_tokens: dict[str, list[str]] = {}
    doc_freq: Counter[str] = Counter()
    for chunk in corpus:
        tokens = _tokenize(chunk_search_text(chunk))
        doc_tokens[chunk["chunk_id"]] = tokens
        doc_freq.update(set(tokens))

    idf = {
        term: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0 for term, freq in doc_freq.items()
    }

    doc_vectors: dict[str, dict[str, float]] = {}
    for chunk_id, tokens in doc_tokens.items():
        if not tokens:
            doc_vectors[chunk_id] = {}
            continue
        counts = Counter(tokens)
        total = float(len(tokens))
        vec = {term: (count / total) * idf[term] for term, count in counts.items()}
        norm = math.sqrt(sum(val * val for val in vec.values()))
        doc_vectors[chunk_id] = (
            {term: val / norm for term, val in vec.items()} if norm > 0.0 else {}
        )
    return idf, doc_vectors


def _tfidf_scores(query_text: str, candidates: list[RagChunk]) -> dict[str, float]:
    tokens = _tokenize(query_text)
    if not tokens:
        return {}
    idf, doc_vectors = _tfidf_index()
    counts = Counter(tokens)
    total = float(len(tokens))
    q_vec = {
        term: (count / total) * idf[term]
        for term, count in counts.items()
        if term in idf
    }
    norm = math.sqrt(sum(val * val for val in q_vec.values()))
    if norm <= 0.0:
        return {}
    q_norm = {term: val / norm for term, val in q_vec.items()}

    scores: dict[str, float] = {}
    for chunk in candidates:
        d_vec = doc_vectors.get(chunk["chunk_id"], {})
        scores[chunk["chunk_id"]] = sum(
            q_val * d_vec[term] for term, q_val in q_norm.items() if term in d_vec
        )
    return scores


@lru_cache(maxsize=64)
def _embed_query(query_text: str, model_name: str, dimensions: int) -> np.ndarray | None:
    """Compute a normalized query embedding via Gemini if an API key is configured."""
    api_key = settings.effective_gemini_api_key
    if not api_key or not model_name:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=5_000),
        )
        config_kwargs: dict[str, Any] = {"task_type": "RETRIEVAL_QUERY"}
        if dimensions > 0:
            config_kwargs["output_dimensionality"] = dimensions

        response = client.models.embed_content(
            model=model_name,
            contents=query_text,
            config=types.EmbedContentConfig(**config_kwargs),
        )
        embeddings = getattr(response, "embeddings", None) or []
        if not embeddings:
            return None
        values = getattr(embeddings[0], "values", None)
        if not values:
            return None
        vec = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else None
    except Exception as exc:  # noqa: BLE001 - network/quota issues must degrade cleanly
        log.debug("Query embedding unavailable (%s); using TF-IDF fallback", exc.__class__.__name__)
        return None


def _vector_scores(query_text: str, candidates: list[RagChunk]) -> dict[str, float]:
    model_name, dimensions, index_vectors = load_index()
    if not index_vectors:
        return {}

    if not any(chunk["chunk_id"] in index_vectors for chunk in candidates):
        return {}

    query_vec = _embed_query(query_text, model_name, dimensions)
    if query_vec is None:
        return {}

    scores: dict[str, float] = {}
    for chunk in candidates:
        doc_vec = index_vectors.get(chunk["chunk_id"])
        if doc_vec is not None and doc_vec.shape == query_vec.shape:
            scores[chunk["chunk_id"]] = float(np.dot(query_vec, doc_vec))
    return scores


def _rank_candidates(candidates: list[RagChunk], query_text: str | None) -> list[RagChunk]:
    if not query_text or not query_text.strip() or len(candidates) <= 1:
        return list(candidates)

    cleaned_query = query_text.strip()
    tfidf = _tfidf_scores(cleaned_query, candidates)
    vec = _vector_scores(cleaned_query, candidates)

    indexed: list[tuple[float, int, RagChunk]] = []
    for pos, chunk in enumerate(candidates):
        cid = chunk["chunk_id"]
        t_score = tfidf.get(cid, 0.0)
        combined = 0.7 * vec[cid] + 0.3 * t_score if cid in vec else t_score
        indexed.append((combined, -pos, chunk))

    indexed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in indexed]


def _make_excerpt(text: str, max_chars: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;।")
    return f"{clipped}..." if clipped else f"{cleaned[:max_chars]}..."


def _dedupe_chemicals(chunks: list[RagChunk]) -> list[RagChemical]:
    seen: set[str] = set()
    items: list[RagChemical] = []
    for chunk in chunks:
        if chunk.get("cultural_only"):
            continue
        for chem in chunk.get("chemicals", []):
            key = chem["active_ingredient"].strip().lower()
            if key and key not in seen:
                seen.add(key)
                items.append(chem)
    return items


def _select_candidates(norm_disease: str, norm_crop: str, corpus: list[RagChunk]) -> list[RagChunk]:
    if norm_disease:
        exact = [
            chunk
            for chunk in corpus
            if chunk["disease_id"] == norm_disease
            and (not norm_crop or norm_crop == "unknown" or chunk["crop_id"] == norm_crop)
        ]
        if exact:
            return exact

        # Fallback if crop hint disagreed or was omitted for a valid disease id
        any_crop = [chunk for chunk in corpus if chunk["disease_id"] == norm_disease]
        if any_crop and (not norm_crop or norm_crop == "unknown"):
            return any_crop

    return []


def _safe_fallback_candidates(norm_crop: str, corpus: list[RagChunk]) -> list[RagChunk]:
    """Return safe non-chemical scouting/monitoring passages for uncertain diagnoses."""
    if norm_crop == "rice":
        rice_scouting = [
            chunk for chunk in corpus if chunk["chunk_id"] == "brri_rice_healthy_02"
        ]
        if rice_scouting:
            return rice_scouting

    if norm_crop == "potato":
        potato_scouting = [
            chunk for chunk in corpus if chunk["chunk_id"] == "bari_potato_healthy_02"
        ]
        if potato_scouting:
            return potato_scouting

    both_scouting = [
        chunk
        for cid in ("brri_rice_healthy_02", "bari_potato_healthy_02")
        for chunk in corpus
        if chunk["chunk_id"] == cid
    ]
    if both_scouting:
        return both_scouting

    return [
        chunk
        for chunk in corpus
        if chunk.get("cultural_only") and not chunk.get("chemicals")
    ]


def retrieve_grounding_bundle(
    disease_id: str,
    crop_id: str,
    query_text: str | None = None,
    top_k: int = 3,
) -> GroundingBundle:
    """Retrieve ranked BARI/BRRI chunks, citations, safety flags, and chemicals."""
    if top_k <= 0:
        return GroundingBundle(
            normalized_disease_id=normalize_disease_id(disease_id),
            sources=[],
            chunks=[],
            cultural_only=False,
            refer_to_officer=False,
            chemicals=[],
            ppe_reminder_bn=None,
            prompt_passages=[],
        )

    corpus = load_corpus()
    norm_disease = normalize_disease_id(disease_id)
    norm_crop = _normalize_crop_id(crop_id)

    used_uncertain_fallback = norm_disease == "uncertain"
    if used_uncertain_fallback:
        candidates = _safe_fallback_candidates(norm_crop, corpus)
    else:
        candidates = _select_candidates(norm_disease, norm_crop, corpus)
        if not candidates:
            used_uncertain_fallback = True
            candidates = _safe_fallback_candidates(norm_crop, corpus)

    selected = _rank_candidates(candidates, query_text)[:top_k]

    cultural_only = used_uncertain_fallback or any(
        chunk.get("cultural_only", False) for chunk in selected
    )
    refer_to_officer = used_uncertain_fallback or any(
        chunk.get("refer_to_officer", False) for chunk in selected
    )
    chemicals = [] if cultural_only else _dedupe_chemicals(selected)

    ppe_reminder_bn = next(
        (chunk["ppe_reminder_bn"] for chunk in selected if chunk.get("ppe_reminder_bn")),
        None,
    )
    if cultural_only:
        ppe_reminder_bn = None

    sources = [
        RagSource(
            chunk_id=chunk["chunk_id"],
            source_title=chunk["source_title"],
            section=chunk["section"],
            disease_id=chunk["disease_id"],
            excerpt=_make_excerpt(chunk["content_bn"]),
            cultural_only=bool(chunk.get("cultural_only", False)),
            refer_to_officer=bool(chunk.get("refer_to_officer", False)),
        )
        for chunk in selected
    ]

    prompt_passages = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_title": chunk["source_title"],
            "section": chunk["section"],
            "title_bn": chunk["title_bn"],
            "content_bn": chunk["content_bn"],
            "cultural_only": bool(chunk.get("cultural_only", False)),
            "refer_to_officer": bool(chunk.get("refer_to_officer", False)),
            "chemicals": (
                [
                    {
                        "bn_name": chem["bn_name"],
                        "dose": chem["dose"],
                        "phi": chem["phi"],
                    }
                    for chem in chunk.get("chemicals", [])
                ]
                if not cultural_only
                else []
            ),
        }
        for chunk in selected
    ]

    if used_uncertain_fallback:
        resolved_id = "uncertain"
    else:
        resolved_id = selected[0]["disease_id"] if selected else norm_disease
    return GroundingBundle(
        normalized_disease_id=resolved_id,
        sources=sources,
        chunks=selected,
        cultural_only=cultural_only,
        refer_to_officer=refer_to_officer,
        chemicals=chemicals,
        ppe_reminder_bn=ppe_reminder_bn,
        prompt_passages=prompt_passages,
    )


def retrieve_grounding(
    disease_id: str,
    crop_id: str,
    query_text: str | None = None,
    top_k: int = 3,
) -> list[RagSource]:
    """Retrieve structured BARI/BRRI source citations for a diagnosed disease."""
    return retrieve_grounding_bundle(
        disease_id=disease_id,
        crop_id=crop_id,
        query_text=query_text,
        top_k=top_k,
    ).sources


def reset_rag_cache() -> None:
    """Clear cached corpus, vector index, and query embeddings (used in tests)."""
    load_corpus.cache_clear()
    load_index.cache_clear()
    _alias_index.cache_clear()
    _tfidf_index.cache_clear()
    _embed_query.cache_clear()
