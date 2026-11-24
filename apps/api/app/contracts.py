"""API contracts.

This module is the source of truth for every request and response shape. FastAPI
derives `openapi.json` from these models, and the frontend types are generated
from that spec, so the client cannot drift from what the server actually serves.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

AnalysisStatus = Literal["pending", "processing", "completed", "failed"]

# Ordered: the UI shows a different line for each.
AnalysisStage = Literal[
    "queued",
    "reading",
    "grounding",
    "writing",
    "speaking",
    "done",
]

ConfidenceBand = Literal["low", "medium", "high"]
AudioStatus = Literal["none", "pending", "ready", "failed"]
GeneratedBy = Literal["llm", "template"]
CropId = Literal["rice", "potato"]
SpraySafety = Literal["green", "amber", "red"]
LocationSource = Literal["voice", "gps", "district"]


class LocationCandidate(BaseModel):
    label: str
    district: str
    latitude: float
    longitude: float


class CropMismatch(BaseModel):
    detected: bool
    photoCropBn: str
    questionCropBn: str
    messageBn: str


class Diagnosis(BaseModel):
    """What the vision model reported. Absent while vision is out of the pipeline."""

    diseaseLabel: str
    cropName: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidenceBand: ConfidenceBand
    isAiEstimate: Literal[True] = True


class Treatment(BaseModel):
    summary: str
    steps: list[str]
    #: The curated source this advice came from, shown to the user.
    source: str | None = None


class RagSource(BaseModel):
    """A grounded BARI/BRRI passage cited for this advisory."""

    chunkId: str
    sourceTitle: str
    section: str
    diseaseId: str
    excerpt: str
    culturalOnly: bool = False
    referToOfficer: bool = False


class Uncertainty(BaseModel):
    isUncertain: bool
    message: str


class HourlySprayWindow(BaseModel):
    timeIso: str
    rainProb: int
    rainMm: float
    windKmh: float
    safety: SpraySafety


class Weather(BaseModel):
    district: str
    #: Rain probability (%) over the next 12 hours.
    rainProbabilityNext12h: int | None = None
    #: Max wind speed (km/h) over the next 12 hours.
    windSpeedMax12h: float | None = None
    humidity: int | None = None
    #: Whether spraying now is sensible, decided in code - never by the LLM.
    sprayWindowOk: bool
    reason: str
    hourlyWindows: list[HourlySprayWindow] = Field(default_factory=list)
    safestWindowSummaryBn: str | None = None
    locationSource: LocationSource = "district"
    resolvedPlace: str | None = None
    gridLatitude: float | None = None
    gridLongitude: float | None = None
    locationCandidates: list[LocationCandidate] = Field(default_factory=list)
    extensionOfficeBn: str | None = None
    source: str = "Open-Meteo"


class AudioInfo(BaseModel):
    status: AudioStatus = "none"
    #: Relative API path, only set when status == "ready".
    url: str | None = None
    durationMs: int | None = None
    byteSize: int | None = None


class AnalysisSummary(BaseModel):
    id: str
    createdAt: datetime
    cropName: str
    status: AnalysisStatus
    stage: AnalysisStage
    thumbnailUrl: str
    imageUrl: str
    diseaseLabel: str | None = None
    district: str | None = None


class AnalysisResult(BaseModel):
    id: str
    createdAt: datetime
    imageUrl: str
    cropName: str
    status: AnalysisStatus
    stage: AnalysisStage
    diagnosis: Diagnosis | None = None
    symptoms: list[str] = Field(default_factory=list)
    treatment: Treatment | None = None
    prevention: list[str] = Field(default_factory=list)
    needsProfessionalVerification: bool = True
    ragSources: list[RagSource] = Field(default_factory=list)
    ppeReminder: str | None = None
    referToOfficer: bool = False
    uncertainty: Uncertainty
    #: Short friendly Bangla text, shown in the UI and spoken by TTS.
    narrative: str = ""
    farmerQuestion: str | None = None
    generatedBy: GeneratedBy = "template"
    weather: Weather | None = None
    cropMismatch: CropMismatch | None = None
    audio: AudioInfo = Field(default_factory=AudioInfo)
    district: str | None = None


class CreateAnalysisResponse(BaseModel):
    id: str
    status: AnalysisStatus
    stage: AnalysisStage
    #: Relative path to poll for progress.
    statusUrl: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm: Literal["ready", "unconfigured"]
    tts: Literal["ready", "unavailable"]
    #: Whether the leaf classifier is enabled. Not a probe of the upstream
    #: model host — the weights download on first use, so "ready" here means
    #: "configured", not "reachable".
    vision: Literal["ready", "unavailable"] = "unavailable"
    weather: Literal["ready", "unknown"]
    db: Literal["ok"]
    version: str


class District(BaseModel):
    id: str
    name: str
    division: str
    latitude: float
    longitude: float


class TtsRequest(BaseModel):
    text: str


class ApiErrorBody(BaseModel):
    code: str
    message: str


class ApiErrorResponse(BaseModel):
    error: ApiErrorBody
