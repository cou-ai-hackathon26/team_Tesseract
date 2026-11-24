"""Leaf disease classification.

One checkpoint: EfficientNet-B0, 8 classes, rice and potato only, loaded from
`VISION_MODEL_DIR` (16 MB, no download). It runs through `onnxruntime`, which
is already a dependency here (it powers the TTS engine), so it adds no install
weight. The PyTorch path the upstream checkpoint ships was not used: it costs a
~900 MB `torch` wheel for a 16 MB model.

The decision rule mirrors `predict_final.py`: restrict the answer to the crop
the farmer selected, choose the best class inside that crop, but judge it on its
raw (unmasked) probability. The masked/renormalised vector is only used for
ranking — thresholding it would let a wrong-crop photo clear the bar, because
renormalisation throws away the evidence that the photo was mostly another crop.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from ..config import settings
from .models import EFFICIENTNET, ModelSpec

log = logging.getLogger(__name__)

VisionStatus = Literal["identified", "undetermined", "unavailable"]

UNMASKED_CROP_MASS_THRESHOLD = 0.60
UNMASKED_TOP_CONFIDENCE_FLOOR = 0.25


class VisionFacts(TypedDict):
    """What the classifier saw. Never asserts a disease on its own."""

    status: VisionStatus
    label: str | None
    #: Softmax probability of the winning class, over the model's full label set.
    confidence: float | None
    crop_id: str | None
    disease_id: str | None
    #: True when the winning class is one of the checkpoint's `*_healthy`
    #: classes. That is a real answer ("the plant looks healthy"), not a miss,
    #: and it must not be dressed up as a disease downstream.
    is_healthy: bool
    #: Which checkpoint produced this, so a stored result stays interpretable
    #: after the checkpoint changes.
    model: str | None
    #: Unmasked global predictions across all classes (before crop masking).
    raw_top_label: NotRequired[str | None]
    raw_top_confidence: NotRequired[float | None]
    inferred_photo_crop: NotRequired[str | None]
    effective_crop: NotRequired[str | None]
    crop_override_detected: NotRequired[bool]
    override_disease_id: NotRequired[str | None]
    override_is_healthy: NotRequired[bool]
    override_confirmed: NotRequired[bool]


class LeafClassifier:
    """Lazy, serialised ONNX inference for the shipped checkpoint.

    `onnxruntime` sessions are thread-safe for `run`, but construction is not,
    so loading takes a lock — the same pattern the TTS engine uses for its
    weights.
    """

    def __init__(self, spec: ModelSpec | None = None) -> None:
        self._spec = spec if spec is not None else EFFICIENTNET
        self._session: Any = None
        self._load_lock = asyncio.Lock()

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def available(self) -> bool:
        return settings.vision_enabled

    @property
    def status(self) -> Literal["ready", "unavailable"]:
        """Whether this stage can actually do anything right now.

        A local checkpoint's bytes either exist or they never will, and
        reporting `ready` for a file that is not there turns every photo into a
        silent "could not identify" instead of a startup error.
        """
        if not settings.vision_enabled:
            return "unavailable"
        if self._session is not None:
            return "ready"
        return "unavailable" if missing_weights(self._spec) else "ready"

    def _load(self) -> Any:
        """Open the ONNX graph from `settings.vision_model_dir`."""
        import onnxruntime as ort

        path = Path(settings.vision_model_dir) / self._spec["file"]
        if not path.exists():
            raise FileNotFoundError(
                f"vision model not found at {path}; put {self._spec['file']} there, "
                "or set VISION_MODEL_DIR to the directory containing it"
            )
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
            sess_options=_session_options(ort),
        )
        log.info(
            "vision model loaded: %s (%d classes)", self._spec["id"], len(self._spec["labels"])
        )
        return session

    async def ensure_loaded(self) -> None:
        if self._session is not None:
            return
        async with self._load_lock:
            if self._session is None:
                self._session = await asyncio.to_thread(self._load)

    async def classify(self, image_path: Path, crop_id: str) -> VisionFacts:
        """Classify one stored image for one crop.

        Never raises: a missing model, unreadable image, or inference error all
        return `unavailable`, because vision is the one stage that may degrade
        without withholding the curated advice behind it.
        """
        if not self.available:
            return _unavailable(self._spec)

        try:
            await self.ensure_loaded()
            return await asyncio.to_thread(self._classify_sync, image_path, crop_id)
        except Exception as exc:  # noqa: BLE001 - vision must never fail the job
            log.warning(
                "vision failed for %s: %s: %s",
                image_path.name,
                exc.__class__.__name__,
                str(exc) or "no detail",
            )
            return _unavailable(self._spec)

    def _classify_sync(self, image_path: Path, crop_id: str) -> VisionFacts:
        from PIL import Image

        spec = self._spec
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB").resize(
                    (spec["input_size"], spec["input_size"]), Image.Resampling.BILINEAR
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("vision image unreadable: %s", exc.__class__.__name__)
            return _unavailable(spec)

        batch = _to_tensor(image, spec)
        logits = self._session.run(None, {spec["input_name"]: batch})[0][0]
        return _decide(_softmax(logits), crop_id, spec)


def missing_weights(spec: ModelSpec) -> Path | None:
    """The local path that is missing, or `None` when this stage can proceed.

    The checkpoint is committed to the repo, so a missing file means a broken
    deployment rather than an optional download. Reporting it from `/health`
    makes that visible before the first upload.
    """
    path = Path(settings.vision_model_dir) / spec["file"]
    return None if path.exists() else path


def _session_options(ort: Any) -> Any:
    options = ort.SessionOptions()
    # One image at a time, two pipeline slots. Extra threads would contend with
    # the TTS synth that shares this process.
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return options


def _to_tensor(image: Any, spec: ModelSpec) -> Any:
    """HWC uint8 -> NCHW float32, rescaled then normalised per the checkpoint.

    Getting mean/std or the input size wrong does not raise; it quietly degrades
    accuracy. That is why they live in the spec next to the label list.
    """
    import numpy as np

    size = spec["input_size"]
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(spec["mean"], dtype=np.float32)
    std = np.array(spec["std"], dtype=np.float32)
    array = (array - mean) / std
    assert array.shape[:2] == (size, size)  # noqa: S101 - preprocessing contract
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def _softmax(logits: Any) -> Any:
    import numpy as np

    shifted = logits - logits.max()
    exponents = np.exp(shifted)
    return exponents / exponents.sum()


def _unmasked_crop_signal(
    probs: Any,
    ranked: Any,
    crop_id: str,
    spec: ModelSpec,
) -> dict[str, Any]:
    """Compute global unmasked softmax crop mass & override metadata across all classes."""
    labels = spec["labels"]
    raw_top_index = int(ranked[0])
    raw_top_label = labels[raw_top_index]
    raw_top_confidence = float(probs[raw_top_index])

    potato_mass = float(
        sum(float(probs[i]) for i, lbl in enumerate(labels) if lbl.startswith("potato_"))
    )
    rice_mass = float(
        sum(float(probs[i]) for i, lbl in enumerate(labels) if lbl.startswith("rice_"))
    )

    inferred_photo_crop: str | None = None
    if (
        potato_mass >= UNMASKED_CROP_MASS_THRESHOLD
        and raw_top_label.startswith("potato_")
        and raw_top_confidence >= UNMASKED_TOP_CONFIDENCE_FLOOR
    ):
        inferred_photo_crop = "potato"
    elif (
        rice_mass >= UNMASKED_CROP_MASS_THRESHOLD
        and raw_top_label.startswith("rice_")
        and raw_top_confidence >= UNMASKED_TOP_CONFIDENCE_FLOOR
    ):
        inferred_photo_crop = "rice"

    crop_override_detected = bool(
        inferred_photo_crop is not None and inferred_photo_crop != crop_id
    )
    override_disease_id: str | None = None
    override_is_healthy = False
    override_confirmed = False

    if crop_override_detected and inferred_photo_crop is not None:
        override_disease_id = spec["label_disease"].get(raw_top_label)
        override_is_healthy = raw_top_label in spec["healthy"]

        inferred_indices = [
            i for i, lbl in enumerate(labels) if spec["label_crop"].get(lbl) == inferred_photo_crop
        ]
        inferred_order = sorted(inferred_indices, key=lambda idx: float(probs[idx]), reverse=True)
        inferred_mass = potato_mass if inferred_photo_crop == "potato" else rice_mass
        runner_up_prob = (
            float(probs[inferred_order[1]]) if len(inferred_order) > 1 else raw_top_confidence
        )
        override_margin = raw_top_confidence - runner_up_prob

        if override_disease_id is not None and (
            raw_top_confidence >= settings.vision_confidence_threshold
            or (
                raw_top_confidence >= settings.vision_margin_floor
                and override_margin >= settings.vision_margin
                and inferred_mass >= settings.vision_margin_min_crop_mass
            )
        ):
            override_confirmed = True

    return {
        "raw_top_label": raw_top_label,
        "raw_top_confidence": raw_top_confidence,
        "inferred_photo_crop": inferred_photo_crop,
        "crop_override_detected": crop_override_detected,
        "override_disease_id": override_disease_id,
        "override_is_healthy": override_is_healthy,
        "override_confirmed": override_confirmed,
    }


def _decide(probabilities: Any, crop_id: str, spec: ModelSpec) -> VisionFacts:
    """Turn probabilities into facts, following `predict_final.py`.

    The winning class is chosen within the selected crop (masked and
    renormalised for ranking only), but its confidence is the raw, unmasked
    softmax probability — the model's actual evidence for that class.

    Two acceptance paths, so a clear photo is not refused just because it falls
    short of one number:

    * primary — raw confidence clears `VISION_CONFIDENCE_THRESHOLD`;
    * margin — raw confidence clears `VISION_MARGIN_FLOOR`, the winner beats the
      runner-up by `VISION_MARGIN`, and the photo is mostly the selected crop
      (`VISION_MARGIN_MIN_CROP_MASS`). This is deliberately stricter about
      ambiguity and crop mass than a flat threshold cut, which on the local test
      sets mostly accepted confident mistakes rather than rescuing clear photos.
    """
    import numpy as np

    labels = spec["labels"]
    probs = np.asarray(probabilities, dtype=np.float64)
    ranked = np.argsort(probs)[::-1]
    top3 = ", ".join(f"{labels[int(i)]}={float(probs[int(i)]):.3f}" for i in ranked[:3])
    unmasked = _unmasked_crop_signal(probs, ranked, crop_id, spec)

    crop_indices = [i for i, label in enumerate(labels) if spec["label_crop"].get(label) == crop_id]
    crop_mass = float(sum(float(probs[i]) for i in crop_indices)) if crop_indices else 0.0

    if crop_indices:
        masked = np.zeros_like(probs)
        masked[crop_indices] = probs[crop_indices]
        total = float(masked.sum())
        masked = masked / total if total > 0.0 else masked
        order = np.argsort(masked)[::-1]
        best_index = int(order[0])
        runner_up_index = int(order[1]) if len(order) > 1 else best_index
    else:
        # Out-of-scope crop: still report the model's own top label, but the
        # acceptance checks below will keep it from becoming a diagnosis.
        best_index = int(ranked[0])
        runner_up_index = int(ranked[1]) if len(ranked) > 1 else best_index

    label = labels[best_index]
    confidence = float(probs[best_index])
    # Raw gap, not the renormalised one: both numbers must be the model's own
    # evidence, or a tiny crop mass can manufacture a huge margin.
    margin = confidence - float(probs[runner_up_index])

    log.info(
        "vision top3 (%s, crop=%s): %s | crop_mass=%.3f best=%s(%.3f) margin=%.3f inferred=%s",
        spec["id"],
        crop_id,
        top3,
        crop_mass,
        label,
        confidence,
        margin,
        unmasked["inferred_photo_crop"],
    )

    if confidence >= settings.vision_confidence_threshold:
        facts = _identified(label, confidence, crop_id, spec)
    elif (
        confidence >= settings.vision_margin_floor
        and margin >= settings.vision_margin
        and crop_mass >= settings.vision_margin_min_crop_mass
    ):
        log.info(
            "vision accepted by margin (%s, crop=%s): %s(%.3f) margin=%.3f crop_mass=%.3f",
            spec["id"],
            crop_id,
            label,
            confidence,
            margin,
            crop_mass,
        )
        facts = _identified(label, confidence, crop_id, spec)
    else:
        facts = _undetermined(label, confidence, spec)

    effective_crop = (
        unmasked["inferred_photo_crop"]
        if unmasked["crop_override_detected"]
        else (facts["crop_id"] or unmasked["inferred_photo_crop"])
    )
    return {
        **facts,
        "raw_top_label": unmasked["raw_top_label"],
        "raw_top_confidence": unmasked["raw_top_confidence"],
        "inferred_photo_crop": unmasked["inferred_photo_crop"],
        "effective_crop": effective_crop,
        "crop_override_detected": unmasked["crop_override_detected"],
        "override_disease_id": unmasked["override_disease_id"],
        "override_is_healthy": unmasked["override_is_healthy"],
        "override_confirmed": unmasked["override_confirmed"],
    }


def _identified(
    label: str, confidence: float, crop_id: str, spec: ModelSpec
) -> VisionFacts:
    """Resolve a winning label to a KB diagnosis or fall back to undetermined."""
    if spec["label_crop"].get(label) != crop_id:
        return _undetermined(label, confidence, spec)
    disease_id = spec["label_disease"].get(label)
    if disease_id is None:
        return _undetermined(label, confidence, spec)
    return VisionFacts(
        status="identified",
        label=label,
        confidence=confidence,
        crop_id=crop_id,
        disease_id=disease_id,
        is_healthy=label in spec["healthy"],
        model=spec["id"],
    )


def _undetermined(label: str | None, confidence: float | None, spec: ModelSpec) -> VisionFacts:
    return VisionFacts(
        status="undetermined",
        label=label,
        confidence=confidence,
        crop_id=None,
        disease_id=None,
        is_healthy=False,
        model=spec["id"],
    )


def _unavailable(spec: ModelSpec | None = None) -> VisionFacts:
    return VisionFacts(
        status="unavailable",
        label=None,
        confidence=None,
        crop_id=None,
        disease_id=None,
        is_healthy=False,
        model=spec["id"] if spec else None,
        raw_top_label=None,
        raw_top_confidence=None,
        inferred_photo_crop=None,
        effective_crop=None,
        crop_override_detected=False,
        override_disease_id=None,
        override_is_healthy=False,
        override_confirmed=False,
    )


_engine: LeafClassifier | None = None


def get_classifier() -> LeafClassifier:
    global _engine
    if _engine is None:
        _engine = LeafClassifier()
    return _engine


def reset_classifier() -> None:
    """Drop the cached model. Used by tests after changing settings."""
    global _engine
    _engine = None
