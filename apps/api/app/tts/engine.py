"""Local Bangla TTS.

Model: `sulabhkatiyar/indic-tiny-tts-bengali` (22.5M params, ONNX, CPU).
Measured on the target machine: ~9-12x faster than real time, ~300 MB peak RSS.

Two known defects in the model's own text normaliser are corrected here before
its input is built, both verified by running it:

    "২ মিলি/লিটার"   ->  "দুই মিলি বা লিটার"      # "/" becomes "or"
    "Tricyclazole"    ->  ""                        # Latin spans deleted

The first turns a dose ratio into nonsense. The second silently removes a
medicine name from a sentence that still sounds complete. Both are handled by
`prepare_text`, and the chemical-name assertion fails loudly rather than
shipping mutilated audio.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
import subprocess
import unicodedata
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
_OPUS_BITRATE = "24k"
_MP3_BITRATE = "32k"

#: Characters that must never reach the model.
_STRIP = str.maketrans(
    {"*": "", "_": "", "`": "", "#": "", ">": "", "[": "", "]": "", "{": "", "}": ""}
)

#: The model deletes Latin spans outright, so any Latin in the text is a hazard.
_LATIN = re.compile(r"[A-Za-z]")


class TtsUnavailable(RuntimeError):
    """The model or the encoder is missing. Text is still served."""


class TtsSafetyError(RuntimeError):
    """Prepared text lost a medicine name. Never ship this as audio."""


@dataclass(frozen=True, slots=True)
class AudioClip:
    path: Path
    duration_ms: int
    byte_size: int
    mime: str


def prepare_text(text: str, required_terms: list[str] | None = None) -> str:
    """Make text safe for this model before it gets its hands on it.

    Runs *before* the model's own normaliser, because two of that normaliser's
    behaviours are unsafe here (see the module docstring).
    """
    prepared = unicodedata.normalize("NFC", text).translate(_STRIP)

    # A slash means "per", not "or". The model reads it as "বা" (or).
    prepared = prepared.replace("/", " প্রতি ")

    # Hyphen ranges read badly aloud; spell the range connector.
    prepared = prepared.replace("—", ", ").replace("–", " থেকে ")
    prepared = prepared.replace("%", " শতাংশ")

    prepared = " ".join(prepared.split())

    # Only check terms that were actually in the input. A narrative names one
    # medicine, not every candidate the knowledge base holds, so demanding all
    # of them would refuse valid audio.
    for term in required_terms or []:
        if term and term in text and term not in prepared:
            raise TtsSafetyError(f"medicine name dropped before synthesis: {term}")

    # The model deletes Latin spans, so any that survived this far would vanish
    # from the audio while the sentence still sounded complete.
    latin = _LATIN.search(prepared)
    if latin:
        raise TtsSafetyError(f"latin text would be dropped by the model: {latin.group(0)}")

    return prepared


def sanitize_for_speech(text: str) -> str:
    """Strip markdown and stray Latin runs (e.g. PPE, GPS) for on-demand UI cards."""
    cleaned = unicodedata.normalize("NFC", text).translate(_STRIP)
    cleaned = re.sub(r"[A-Za-z]+", " ", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return " ".join(cleaned.split()).strip()


def _encoder_args(target: Path) -> tuple[list[str], str]:
    if settings.tts_format == "opus":
        return (
            ["-c:a", "libopus", "-b:a", _OPUS_BITRATE, "-ac", "1", "-ar", str(SAMPLE_RATE)],
            "audio/ogg",
        )
    return (
        ["-c:a", "libmp3lame", "-b:a", _MP3_BITRATE, "-ac", "1", "-ar", "22050"],
        "audio/mpeg",
    )


def _encode(wav_path: Path, target: Path) -> int:
    """Transcode to Opus/MP3. Returns the byte size.

    A 7.8s narrative is 367 KB as WAV and 22 KB as Opus. On rural mobile data
    that is the difference between usable and not.
    """
    ffmpeg = shutil.which(settings.ffmpeg_path)
    if not ffmpeg:
        raise TtsUnavailable("ffmpeg not found")

    args, _ = _encoder_args(target)
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path), *args, str(target)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise TtsUnavailable(f"ffmpeg failed: {result.stderr.decode(errors='replace')[:200]}")

    return target.stat().st_size


def _has_ffmpeg() -> bool:
    return shutil.which(settings.ffmpeg_path) is not None


def _effective_format() -> str:
    return settings.tts_format if _has_ffmpeg() else "wav"


def _write_wav(samples: Any, dest: Path) -> None:
    import numpy as np

    tmp_path = dest.with_name(f"{dest.name}.tmp")
    clipped = np.clip(np.asarray(samples), -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(str(tmp_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    tmp_path.replace(dest)


class TtsEngine:
    def __init__(self) -> None:
        self._tts: object | None = None
        self._load_lock = asyncio.Lock()
        #: Synthesis is serialised on purpose. It is the heaviest stage on a
        #: CPU-only box with ~3 GiB free; running two at once makes both slower
        #: and risks the OOM killer taking the API down mid-request.
        self._synth_lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return settings.tts_enabled

    def _load(self) -> object:
        """Download (once) and load the ONNX model. Blocking."""
        import sys

        from huggingface_hub import snapshot_download

        folder = snapshot_download(
            settings.tts_model_id,
            ignore_patterns=["*.safetensors", "samples/*", "assets/*"],
        )
        if folder not in sys.path:
            sys.path.insert(0, folder)
        from indic_tiny_tts import TinyTTSOnnx  # type: ignore[import-not-found]

        return TinyTTSOnnx.from_pretrained(folder)

    async def ensure_loaded(self) -> None:
        if self._tts is not None:
            return
        async with self._load_lock:
            if self._tts is None:
                self._tts = await asyncio.to_thread(self._load)
                log.info("TTS model loaded: %s", settings.tts_model_id)

    async def synthesise(
        self, text: str, out_dir: Path, required_terms: list[str] | None = None
    ) -> AudioClip:
        """Synthesise to a cached file. Blocking work runs off the event loop."""
        out_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_text(text, required_terms)
        fmt = _effective_format()
        digest = hashlib.sha256(
            f"{prepared}|{settings.tts_voice}|{fmt}".encode()
        ).hexdigest()[:24]

        target = out_dir / f"{digest}.{fmt}"
        if target.exists():
            return AudioClip(
                target, _duration_ms(target), target.stat().st_size, audio_mime(target)
            )

        await self.ensure_loaded()
        async with self._synth_lock:
            # Re-check inside the lock: another request may have produced the
            # same clip while this one waited.
            if target.exists():
                return AudioClip(
                    target, _duration_ms(target), target.stat().st_size, audio_mime(target)
                )
            return await asyncio.to_thread(self._synthesise_blocking, prepared, out_dir, target)

    def _synthesise_blocking(self, prepared: str, out_dir: Path, target: Path) -> AudioClip:
        if self._tts is None:
            # A real check, not an assert: `python -O` strips asserts.
            raise TtsUnavailable("TTS model not loaded")
        wav = self._tts.synthesize(prepared, voice=settings.tts_voice)  # type: ignore[attr-defined]

        if target.suffix == ".wav":
            _write_wav(wav, target)
            log.info("ffmpeg not found; generated WAV audio clip %s", target.name)
            return AudioClip(
                target, _duration_ms(target), target.stat().st_size, audio_mime(target)
            )

        raw = out_dir / f"{target.stem}.wav"
        _write_wav(wav, raw)
        try:
            size = _encode(raw, target)
        except TtsUnavailable as exc:
            log.warning("ffmpeg encoding failed (%s); falling back to WAV %s", exc, raw.name)
            return AudioClip(raw, _duration_ms(raw), raw.stat().st_size, audio_mime(raw))
        raw.unlink(missing_ok=True)

        return AudioClip(target, _duration_ms(target), size, audio_mime(target))


def audio_mime(path: Path) -> str:
    if path.suffix == ".opus":
        return "audio/ogg"
    if path.suffix == ".wav":
        return "audio/wav"
    return "audio/mpeg"


_mime = audio_mime


def _duration_ms(path: Path) -> int:
    if path.suffix == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                return round((handle.getnframes() / rate) * 1000) if rate > 0 else 0
        except (wave.Error, OSError, EOFError):
            return 0
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    result = subprocess.run(  # noqa: S603
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return round(float(result.stdout.strip()) * 1000)
    except ValueError:
        return 0


@lru_cache
def get_engine() -> TtsEngine:
    return TtsEngine()
