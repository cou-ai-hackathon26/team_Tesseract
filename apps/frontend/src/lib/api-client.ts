import type {
  AnalysisResult,
  AnalysisStage,
  AnalysisStatus,
  AnalysisSummary,
  Weather,
} from '@foshol-lens/contracts';

const RAW_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').trim();
const BASE_URL = (RAW_BASE_URL || 'http://localhost:4000').replace(/\/+$/, '');
const API_PREFIX = '/api/v1';

/** Poll interval while an analysis is running. */
export const POLL_INTERVAL_MS = 700;

/** Give up after this long rather than polling forever. */
const POLL_TIMEOUT_MS = 90_000;

export interface CreateAnalysisInput {
  /** Downscaled image produced by prepareImage(). */
  imageBlob: Blob;
  crop: string;
  district: string | null;
  latitude?: number | null;
  longitude?: number | null;
  questionText?: string | null;
  audioBlob?: Blob | null;
}

export interface Created {
  id: string;
  status: AnalysisStatus;
  stage: AnalysisStage;
}

/** A district as the API returns it. */
export interface District {
  id: string;
  name: string;
  division: string;
  latitude: number;
  longitude: number;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message?: string,
  ) {
    super(message ?? code);
    this.name = 'ApiError';
  }
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${API_PREFIX}${path}`, init);
  } catch {
    throw new ApiError(0, 'network_error', 'সার্ভারে পৌঁছানো যায়নি');
  }

  if (!response.ok) {
    let code = `http_${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (typeof body.detail === 'string') code = body.detail;
    } catch {
      // Non-JSON error body; keep the status-derived code.
    }
    throw new ApiError(response.status, code);
  }

  return response;
}

/** Resolve relative media paths (`/api/v1/...`) against the API origin. */
export function mediaUrl(path: string | null | undefined): string | undefined {
  if (!path) return undefined;
  if (path.startsWith('http') || path.startsWith('data:') || path.startsWith('blob:')) return path;
  return `${BASE_URL}${path}`;
}

function audioFilenameForMime(mimeType: string): string {
  const base = (mimeType.split(';', 1)[0] ?? '').trim().toLowerCase();
  if (base === 'audio/ogg') return 'question.ogg';
  if (base === 'audio/mp4') return 'question.m4a';
  if (base === 'audio/mpeg') return 'question.mp3';
  if (base === 'audio/wav') return 'question.wav';
  return 'question.webm';
}

export const api = {
  async createAnalysis({
    imageBlob,
    crop,
    district,
    latitude,
    longitude,
    questionText,
    audioBlob,
  }: CreateAnalysisInput): Promise<Created> {
    const body = new FormData();
    body.append('image', imageBlob, 'crop.jpg');
    body.append('crop', crop);
    if (district) body.append('district', district);
    if (
      typeof latitude === 'number' &&
      Number.isFinite(latitude) &&
      typeof longitude === 'number' &&
      Number.isFinite(longitude)
    ) {
      body.append('latitude', String(latitude));
      body.append('longitude', String(longitude));
    }
    const trimmedQuestion = questionText?.trim();
    if (trimmedQuestion) body.append('question_text', trimmedQuestion);
    if (audioBlob && audioBlob.size > 0) {
      body.append('audio', audioBlob, audioFilenameForMime(audioBlob.type));
    }

    const response = await request('/analyses', { method: 'POST', body });
    return (await response.json()) as Created;
  },

  async synthesizeSpeech(text: string): Promise<Blob> {
    const response = await request('/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    return response.blob();
  },

  async listAnalyses(): Promise<AnalysisSummary[]> {
    const response = await request('/analyses');
    return (await response.json()) as AnalysisSummary[];
  },

  async listDistricts(): Promise<District[]> {
    const response = await request('/districts');
    return (await response.json()) as District[];
  },

  async getWeather(lat: number, lon: number, label?: string | null): Promise<Weather> {
    const params = new URLSearchParams({
      lat: String(lat),
      lon: String(lon),
    });
    const trimmedLabel = label?.trim();
    if (trimmedLabel) params.set('label', trimmedLabel);
    const response = await request(`/weather?${params.toString()}`);
    return (await response.json()) as Weather;
  },

  async getSummary(id: string): Promise<AnalysisSummary> {
    const response = await request(`/analyses/${encodeURIComponent(id)}`);
    return (await response.json()) as AnalysisSummary;
  },

  async getResult(id: string): Promise<AnalysisResult> {
    const response = await request(`/analyses/${encodeURIComponent(id)}/result`);
    return (await response.json()) as AnalysisResult;
  },

  /**
   * Poll until the result exists. Resolves with the finished analysis, and calls
   * `onStage` on every change so the UI can describe what is happening.
   */
  async waitForResult(
    id: string,
    onStage?: (stage: AnalysisStage, status: AnalysisStatus) => void,
  ): Promise<AnalysisResult> {
    const startedAt = Date.now();
    let lastStage: AnalysisStage | null = null;

    for (;;) {
      if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
        throw new ApiError(0, 'timeout', 'বিশ্লেষণ শেষ হয়নি');
      }

      const summary = await api.getSummary(id);
      if (summary.stage !== lastStage) {
        lastStage = summary.stage;
        onStage?.(summary.stage, summary.status);
      }

      if (summary.status === 'failed') {
        throw new ApiError(0, 'analysis_failed', 'বিশ্লেষণ ব্যর্থ হয়েছে');
      }

      if (summary.status === 'completed') {
        return api.getResult(id);
      }

      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    }
  },
};

export type Api = typeof api;
