"""Build precomputed Gemini embeddings for the BARI/BRRI RAG corpus.

Run from `apps/api`:
    .venv/Scripts/python.exe scripts/build_rag_index.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.config import settings  # noqa: E402
from app.grounding.rag import (  # noqa: E402
    DEFAULT_CORPUS_PATH,
    DEFAULT_INDEX_PATH,
    RagChunk,
    chunk_search_text,
)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
FALLBACK_EMBEDDING_MODELS: tuple[str, ...] = (
    "gemini-embedding-001",
    "text-embedding-004",
)


def _candidate_models(preferred: str) -> list[str]:
    models = [preferred]
    for fallback in FALLBACK_EMBEDDING_MODELS:
        if fallback not in models:
            models.append(fallback)
    return models


def _embed_batch(
    client: Any,
    model_name: str,
    texts: list[str],
    dims: int,
) -> tuple[str, list[list[float]]]:
    from google.genai import errors as genai_errors
    from google.genai import types

    last_error: Exception | None = None
    for candidate in _candidate_models(model_name):
        try:
            config_kwargs: dict[str, Any] = {"task_type": "RETRIEVAL_DOCUMENT"}
            if dims > 0:
                config_kwargs["output_dimensionality"] = dims
            response = client.models.embed_content(
                model=candidate,
                contents=texts,
                config=types.EmbedContentConfig(**config_kwargs),
            )
            vectors: list[list[float]] = []
            for item in response.embeddings:
                arr = np.asarray(item.values, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 0.0:
                    arr = arr / norm
                vectors.append([round(float(v), 6) for v in arr.tolist()])
            return candidate, vectors
        except genai_errors.APIError as exc:
            last_error = exc
            if exc.code == 404:
                continue
            raise

    raise RuntimeError(f"No supported embedding model succeeded: {last_error}") from last_error


def build_index(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_path: Path = DEFAULT_INDEX_PATH,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    dims: int = 768,
) -> int:
    """Compute normalized Gemini embeddings for every chunk and write index JSON."""
    from google import genai

    api_key = settings.effective_gemini_api_key
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or LLM_API_KEY) is required to build the vector index."
        )

    corpus: list[RagChunk] = json.loads(corpus_path.read_text(encoding="utf-8"))
    texts = [chunk_search_text(chunk) for chunk in corpus]

    client = genai.Client(api_key=api_key)
    used_model, vectors = _embed_batch(client, model_name, texts, dims)
    if len(vectors) != len(corpus):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(corpus)}, got {len(vectors)}"
        )

    actual_dims = len(vectors[0]) if vectors else dims
    index_payload = {
        "model": used_model,
        "dimensions": actual_dims,
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_file": corpus_path.name,
        "embeddings": {
            chunk["chunk_id"]: vector for chunk, vector in zip(corpus, vectors, strict=True)
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(corpus)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BARI/BRRI RAG vector index.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dims", type=int, default=768)
    args = parser.parse_args()

    count = build_index(
        corpus_path=args.corpus,
        output_path=args.output,
        model_name=args.model,
        dims=args.dims,
    )
    print(f"Indexed {count} BARI/BRRI chunks -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
