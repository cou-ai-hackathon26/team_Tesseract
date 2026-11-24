/**
 * Canonical API route paths. The backend mounts these under /api/v1 and the
 * frontend API client builds its URLs from them - never hardcode a path.
 */
export const API_PREFIX = '/api/v1' as const;

export const API_ROUTES = {
  health: '/health',
  analyses: '/analyses',
  weather: '/weather',
  tts: '/tts',
} as const;

export function analysisPath(id: string): string {
  return `${API_ROUTES.analyses}/${encodeURIComponent(id)}`;
}

export function analysisResultPath(id: string): string {
  return `${analysisPath(id)}/result`;
}

export function apiUrl(path: string): string {
  return `${API_PREFIX}${path}`;
}
