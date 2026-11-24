import { useEffect, useState } from 'react';
import { api, type District } from './api-client';

const STORAGE_KEY = 'fasol-lens.district.v1';

export interface DistrictOption {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
}

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const rKm = 6371;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return 2 * rKm * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function findNearestDistrict(
  latitude: number,
  longitude: number,
  districts: DistrictOption[],
): DistrictOption | null {
  let nearest: DistrictOption | null = null;
  let minDistance = Infinity;
  for (const item of districts) {
    const distance = haversineKm(latitude, longitude, item.latitude, item.longitude);
    if (distance < minDistance) {
      minDistance = distance;
      nearest = item;
    }
  }
  return nearest;
}

/**
 * Remember the chosen district across visits and support opt-in GPS matching.
 */
export function useDistrict() {
  const [district, setDistrict] = useState<string>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) ?? '';
    } catch {
      return '';
    }
  });

  const [districts, setDistricts] = useState<DistrictOption[]>([]);

  useEffect(() => {
    let cancelled = false;
    api
      .listDistricts()
      .then((list: District[]) => {
        if (!cancelled) {
          setDistricts(
            list.map(({ id, name, latitude, longitude }) => ({
              id,
              name,
              latitude,
              longitude,
            })),
          );
        }
      })
      .catch(() => {
        // Leave the selector empty; weather is optional.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function choose(next: string) {
    setDistrict(next);
    try {
      if (next) localStorage.setItem(STORAGE_KEY, next);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Private mode: the in-memory value still works for this session.
    }
  }

  return { district, districts, choose };
}
