"""Precise Bangladesh village/plot geocoding with 3-step disambiguation.

Layer 1 (Primary): Open-Meteo Geocoding API (GeoNames-backed, BD filter).
Layer 2 (Fallback): OpenStreetMap Nominatim for rural villages/unions absent in GeoNames.
Layer 3 (Offline): Local 64-district table (+ key rural upazilas) so resolution never fails offline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .districts import DISTRICTS, _haversine_km, find_nearest_district, get_district

log = logging.getLogger(__name__)

OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
GEOCODE_USER_AGENT = "FosholLens-CoU-Hackathon/1.0"
GEOCODE_TIMEOUT_SECONDS = 4.0

_ADMIN_NOISE_RE = re.compile(r"\b(?:district|zila|zilla|division|sadar|upazila)\b", re.IGNORECASE)

_DISTRICT_ALIASES: dict[str, str] = {
    "cumilla": "comilla",
    "comilla": "comilla",
    "bogura": "bogura",
    "bogra": "bogura",
    "chattogram": "chattogram",
    "chittagong": "chattogram",
    "jashore": "jashore",
    "jessore": "jashore",
    "barishal": "barishal",
    "barisal": "barishal",
    "coxs bazar": "coxsbazar",
    "cox's bazar": "coxsbazar",
    "coxsbazar": "coxsbazar",
    "chapainawabganj": "chapainawabganj",
    "chapai nawabganj": "chapainawabganj",
    "nawabganj": "chapainawabganj",
    "netrakona": "netrokona",
    "netrokona": "netrokona",
    "khagrachari": "khagrachhari",
    "khagrachhari": "khagrachhari",
    "jhalokati": "jhalokathi",
    "jhalokathi": "jhalokathi",
    "brahmanbaria": "brahmanbaria",
    "b baria": "brahmanbaria",
    "maulvibazar": "moulvibazar",
    "moulvibazar": "moulvibazar",
    "joypurhat": "joypurhat",
    "jaipurhat": "joypurhat",
    "chuadanga": "chuadanga",
    "jhenaidah": "jhenaidah",
    "kishoreganj": "kishoreganj",
    "shariatpur": "shariatpur",
    "munshiganj": "munshiganj",
    "narayanganj": "narayanganj",
    "habiganj": "habiganj",
    "sunamganj": "sunamganj",
    "sirajganj": "sirajganj",
    "manikganj": "manikganj",
    "gopalganj": "gopalganj",
    "lakshmipur": "lakshmipur",
    "laxmipur": "lakshmipur",
    "chapai": "chapainawabganj",
}

for _district in DISTRICTS:
    _did = str(_district["id"])
    _DISTRICT_ALIASES[_did] = _did
    _DISTRICT_ALIASES[str(_district["name"]).strip().lower()] = _did

_DIVISION_EN_TO_BN: dict[str, str] = {
    "dhaka": "ঢাকা",
    "chattogram": "চট্টগ্রাম",
    "chittagong": "চট্টগ্রাম",
    "rajshahi": "রাজশাহী",
    "khulna": "খুলনা",
    "barishal": "বরিশাল",
    "barisal": "বরিশাল",
    "sylhet": "সিলেট",
    "rangpur": "রংপুর",
    "mymensingh": "ময়মনসিংহ",
}

#: Offline rural sub-district coordinates used only when Layers 1 & 2 are unreachable.
_OFFLINE_PLACES: dict[str, list[dict[str, Any]]] = {
    "burichang": [
        {
            "name_en": "Burichang",
            "name_bn": "বুড়িচং",
            "district_id": "comilla",
            "latitude": 23.5494,
            "longitude": 91.1242,
        }
    ],
    "debidwar": [
        {
            "name_en": "Debidwar",
            "name_bn": "দেবিদ্বার",
            "district_id": "comilla",
            "latitude": 23.6003,
            "longitude": 90.9854,
        }
    ],
    "daudkandi": [
        {
            "name_en": "Daudkandi",
            "name_bn": "দাউদকান্দি",
            "district_id": "comilla",
            "latitude": 23.5333,
            "longitude": 90.7167,
        }
    ],
    "chauddagram": [
        {
            "name_en": "Chauddagram",
            "name_bn": "চৌদ্দগ্রাম",
            "district_id": "comilla",
            "latitude": 23.2266,
            "longitude": 91.3189,
        }
    ],
    "sherpur": [
        {
            "name_en": "Sherpur",
            "name_bn": "শেরপুর",
            "district_id": "bogura",
            "latitude": 24.6749,
            "longitude": 89.4170,
        },
        {
            "name_en": "Sherpur",
            "name_bn": "শেরপুর",
            "district_id": "sherpur",
            "latitude": 25.0205,
            "longitude": 90.0153,
        },
        {
            "name_en": "Sherpur",
            "name_bn": "শেরপুর",
            "district_id": "moulvibazar",
            "latitude": 24.6272,
            "longitude": 91.6817,
        },
    ],
    "kaliganj": [
        {
            "name_en": "Kaliganj",
            "name_bn": "কালীগঞ্জ",
            "district_id": "gazipur",
            "latitude": 23.9255,
            "longitude": 90.5669,
        },
        {
            "name_en": "Kaliganj",
            "name_bn": "কালীগঞ্জ",
            "district_id": "jhenaidah",
            "latitude": 23.4126,
            "longitude": 89.1381,
        },
        {
            "name_en": "Kaliganj",
            "name_bn": "কালীগঞ্জ",
            "district_id": "satkhira",
            "latitude": 22.4488,
            "longitude": 89.0372,
        },
    ],
    "shibganj": [
        {
            "name_en": "Shibganj",
            "name_bn": "শিবগঞ্জ",
            "district_id": "bogura",
            "latitude": 25.0029,
            "longitude": 89.3177,
        },
        {
            "name_en": "Shibganj",
            "name_bn": "শিবগঞ্জ",
            "district_id": "chapainawabganj",
            "latitude": 24.6844,
            "longitude": 88.1583,
        },
    ],
    "savar": [
        {
            "name_en": "Savar",
            "name_bn": "সাভার",
            "district_id": "dhaka",
            "latitude": 23.8583,
            "longitude": 90.2667,
        }
    ],
    "pirganj": [
        {
            "name_en": "Pirganj",
            "name_bn": "পীরগঞ্জ",
            "district_id": "rangpur",
            "latitude": 25.4112,
            "longitude": 89.3173,
        },
        {
            "name_en": "Pirganj",
            "name_bn": "পীরগঞ্জ",
            "district_id": "thakurgaon",
            "latitude": 25.8548,
            "longitude": 88.3669,
        },
    ],
    "companiganj": [
        {
            "name_en": "Companiganj",
            "name_bn": "কোম্পানীগঞ্জ",
            "district_id": "noakhali",
            "latitude": 22.8758,
            "longitude": 91.2856,
        },
        {
            "name_en": "Companiganj",
            "name_bn": "কোম্পানীগঞ্জ",
            "district_id": "sylhet",
            "latitude": 25.0750,
            "longitude": 91.7583,
        },
    ],
}

_OFFLINE_PLACES_BY_BN: dict[str, str] = {
    str(entry["name_bn"]): en_key
    for en_key, entries in _OFFLINE_PLACES.items()
    for entry in entries
    if entry.get("name_bn")
}


@dataclass(frozen=True, slots=True)
class LocationCandidateInfo:
    label: str
    district: str
    latitude: float
    longitude: float

    def as_contract_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "district": self.district,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
        }


@dataclass(frozen=True, slots=True)
class ResolvedLocation:
    resolved_place: str
    district_id: str
    district_name: str
    latitude: float
    longitude: float
    candidates: tuple[LocationCandidateInfo, ...]
    source_layer: str


@dataclass(slots=True)
class _RawCandidate:
    place_label: str
    district_id: str
    district_name: str
    division_name: str
    latitude: float
    longitude: float
    admin1: str
    admin2: str
    rank: int


def _normalize_district_id(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = _ADMIN_NOISE_RE.sub(" ", raw).strip().lower()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None
    if cleaned in _DISTRICT_ALIASES:
        return _DISTRICT_ALIASES[cleaned]
    for alias, did in _DISTRICT_ALIASES.items():
        if alias and alias in cleaned:
            return did
    return None


def _normalize_division_bn(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = _ADMIN_NOISE_RE.sub(" ", raw).strip().lower()
    for key, bn in _DIVISION_EN_TO_BN.items():
        if key in cleaned:
            return bn
    return None


def _format_candidate_label(
    *,
    place_bn: str | None,
    place_en: str | None,
    raw_name: str | None,
    district_name: str,
    division_name: str,
) -> tuple[str, str]:
    base_place = (place_bn or "").strip() or (raw_name or "").strip() or (place_en or "").strip()
    if not base_place:
        base_place = district_name

    if base_place == district_name:
        if division_name and division_name != district_name:
            return f"{base_place}, {division_name}", division_name
        return base_place, district_name
    return f"{base_place}, {district_name}", district_name


def _resolve_admin_district(
    lat: float,
    lon: float,
    *admin_hints: str | None,
) -> dict[str, object]:
    for hint in admin_hints:
        did = _normalize_district_id(hint)
        if did:
            found = get_district(did)
            if found is not None:
                return found
    return find_nearest_district(lat, lon)


async def _query_open_meteo(query: str) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "name": query,
        "count": 5,
        "language": "en",
        "format": "json",
        "countryCode": "BD",
    }
    try:
        async with httpx.AsyncClient(timeout=GEOCODE_TIMEOUT_SECONDS) as client:
            resp = await client.get(OPEN_METEO_GEOCODE_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Open-Meteo geocoding failed for %r: %s", query, exc.__class__.__name__)
        return []

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    return [
        item
        for item in results
        if isinstance(item, dict)
        and str(item.get("country_code") or "BD").upper() == "BD"
        and item.get("latitude") is not None
        and item.get("longitude") is not None
    ]


async def _query_nominatim(query: str) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "q": f"{query}, Bangladesh",
        "format": "json",
        "limit": 5,
        "countrycodes": "bd",
        "addressdetails": 1,
    }
    headers = {"User-Agent": GEOCODE_USER_AGENT}
    try:
        async with httpx.AsyncClient(timeout=GEOCODE_TIMEOUT_SECONDS, headers=headers) as client:
            resp = await client.get(NOMINATIM_SEARCH_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Nominatim geocoding failed for %r: %s", query, exc.__class__.__name__)
        return []

    if not isinstance(payload, list):
        return []
    return [
        item
        for item in payload
        if isinstance(item, dict) and item.get("lat") is not None and item.get("lon") is not None
    ]


def _normalize_open_meteo_results(
    results: list[dict[str, Any]],
    place_bn: str | None,
    place_en: str | None,
) -> list[_RawCandidate]:
    out: list[_RawCandidate] = []
    for idx, item in enumerate(results):
        try:
            lat = float(item["latitude"])
            lon = float(item["longitude"])
        except (TypeError, ValueError):
            continue
        admin1 = str(item.get("admin1") or "")
        admin2 = str(item.get("admin2") or "")
        raw_name = str(item.get("name") or "")
        district_row = _resolve_admin_district(lat, lon, admin2, raw_name)
        district_id = str(district_row["id"])
        district_name = str(district_row["name"])
        division_name = str(district_row["division"])
        label, display_district = _format_candidate_label(
            place_bn=place_bn,
            place_en=place_en,
            raw_name=raw_name,
            district_name=district_name,
            division_name=division_name,
        )
        out.append(
            _RawCandidate(
                place_label=label,
                district_id=district_id,
                district_name=display_district,
                division_name=division_name,
                latitude=lat,
                longitude=lon,
                admin1=admin1,
                admin2=admin2,
                rank=idx,
            )
        )
    return out


def _normalize_nominatim_results(
    results: list[dict[str, Any]],
    place_bn: str | None,
    place_en: str | None,
) -> list[_RawCandidate]:
    out: list[_RawCandidate] = []
    for idx, item in enumerate(results):
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except (TypeError, ValueError):
            continue
        raw_address = item.get("address")
        address: dict[str, Any] = raw_address if isinstance(raw_address, dict) else {}
        admin2 = str(
            address.get("state_district")
            or address.get("county")
            or address.get("city")
            or ""
        )
        admin1 = str(address.get("state") or "")
        raw_name = str(
            item.get("name")
            or address.get("village")
            or address.get("town")
            or address.get("suburb")
            or address.get("hamlet")
            or ""
        )
        if not raw_name and item.get("display_name"):
            raw_name = str(item["display_name"]).split(",", 1)[0].strip()

        district_row = _resolve_admin_district(lat, lon, admin2, raw_name)
        district_id = str(district_row["id"])
        district_name = str(district_row["name"])
        division_name = str(district_row["division"])
        label, display_district = _format_candidate_label(
            place_bn=place_bn,
            place_en=place_en,
            raw_name=raw_name,
            district_name=district_name,
            division_name=division_name,
        )
        out.append(
            _RawCandidate(
                place_label=label,
                district_id=district_id,
                district_name=display_district,
                division_name=division_name,
                latitude=lat,
                longitude=lon,
                admin1=admin1,
                admin2=admin2,
                rank=idx,
            )
        )
    return out


def _offline_candidates(
    place_en: str | None,
    place_bn: str | None,
    spoken_district_en: str | None,
) -> list[_RawCandidate]:
    out: list[_RawCandidate] = []
    key_en = (place_en or "").strip().lower()
    if key_en not in _OFFLINE_PLACES and place_bn:
        key_en = _OFFLINE_PLACES_BY_BN.get(place_bn.strip(), key_en)
    if key_en and key_en in _OFFLINE_PLACES:
        for idx, entry in enumerate(_OFFLINE_PLACES[key_en]):
            district_row = get_district(str(entry["district_id"]))
            if district_row is None:
                continue
            district_id = str(district_row["id"])
            district_name = str(district_row["name"])
            division_name = str(district_row["division"])
            label, display_district = _format_candidate_label(
                place_bn=place_bn or str(entry["name_bn"]),
                place_en=place_en or str(entry["name_en"]),
                raw_name=str(entry["name_en"]),
                district_name=district_name,
                division_name=division_name,
            )
            out.append(
                _RawCandidate(
                    place_label=label,
                    district_id=district_id,
                    district_name=display_district,
                    division_name=division_name,
                    latitude=float(entry["latitude"]),
                    longitude=float(entry["longitude"]),
                    admin1=division_name,
                    admin2=district_id,
                    rank=idx,
                )
            )
        if out:
            return out

    seen_ids: set[str] = set()
    for idx, hint in enumerate((spoken_district_en, place_en, place_bn)):
        did = _normalize_district_id(hint)
        if not did or did in seen_ids:
            continue
        district_row = get_district(did)
        if district_row is None:
            continue
        seen_ids.add(did)
        district_name = str(district_row["name"])
        division_name = str(district_row["division"])
        label = (
            f"{place_bn}, {district_name}"
            if place_bn and place_bn.strip() and place_bn.strip() != district_name
            else district_name
        )
        out.append(
            _RawCandidate(
                place_label=label,
                district_id=did,
                district_name=district_name,
                division_name=division_name,
                latitude=float(str(district_row["latitude"])),
                longitude=float(str(district_row["longitude"])),
                admin1=division_name,
                admin2=did,
                rank=idx,
            )
        )
    return out


def _dedupe_candidates(candidates: list[_RawCandidate]) -> list[_RawCandidate]:
    kept: list[_RawCandidate] = []
    for cand in candidates:
        duplicate = any(
            existing.district_id == cand.district_id
            and _haversine_km(existing.latitude, existing.longitude, cand.latitude, cand.longitude)
            < 15.0
            for existing in kept
        )
        if not duplicate:
            kept.append(cand)
    return kept


def _matches_spoken_district(cand: _RawCandidate, spoken_district_en: str | None) -> bool:
    if not spoken_district_en:
        return False
    target_id = _normalize_district_id(spoken_district_en)
    if target_id and cand.district_id == target_id:
        return True
    target_div_bn = _normalize_division_bn(spoken_district_en)
    if target_div_bn and cand.division_name == target_div_bn:
        return True
    cleaned_target = spoken_district_en.strip().lower()
    return bool(
        cleaned_target
        and (cleaned_target in cand.admin2.lower() or cleaned_target in cand.admin1.lower())
    )


def _tie_break_coords(
    gps_lat: float | None,
    gps_lon: float | None,
    fallback_district_id: str | None,
) -> tuple[float, float] | None:
    if gps_lat is not None and gps_lon is not None:
        return gps_lat, gps_lon
    if fallback_district_id:
        row = get_district(fallback_district_id)
        if row is not None:
            return float(str(row["latitude"])), float(str(row["longitude"]))
    return None


async def resolve_precise_location(
    place_en: str | None,
    place_bn: str | None,
    spoken_district_en: str | None = None,
    gps_lat: float | None = None,
    gps_lon: float | None = None,
    fallback_district_id: str | None = None,
) -> ResolvedLocation | None:
    """Resolve a voice-named village/plot using Open-Meteo -> Nominatim -> offline districts."""
    query = (
        (place_en or "").strip()
        or (place_bn or "").strip()
        or (spoken_district_en or "").strip()
    )
    if not query:
        return None

    source_layer = "open-meteo"
    raw_candidates: list[_RawCandidate] = []

    om_results = await _query_open_meteo(query)
    if om_results:
        raw_candidates = _normalize_open_meteo_results(om_results, place_bn, place_en)

    if not raw_candidates:
        source_layer = "nominatim"
        nom_results = await _query_nominatim(query)
        if nom_results:
            raw_candidates = _normalize_nominatim_results(nom_results, place_bn, place_en)

    if not raw_candidates:
        source_layer = "offline"
        raw_candidates = _offline_candidates(place_en, place_bn, spoken_district_en)

    if not raw_candidates:
        return None

    deduped = _dedupe_candidates(raw_candidates)
    ref_coords = _tie_break_coords(gps_lat, gps_lon, fallback_district_id)

    def _sort_key(cand: _RawCandidate) -> tuple[int, float, int]:
        district_tier = 0 if _matches_spoken_district(cand, spoken_district_en) else 1
        dist_km = (
            _haversine_km(cand.latitude, cand.longitude, ref_coords[0], ref_coords[1])
            if ref_coords is not None
            else 0.0
        )
        return (district_tier, dist_km, cand.rank)

    ranked = sorted(deduped, key=_sort_key)
    top = ranked[0]

    candidate_infos: tuple[LocationCandidateInfo, ...]
    if len(ranked) > 1:
        candidate_infos = tuple(
            LocationCandidateInfo(
                label=item.place_label,
                district=item.district_name,
                latitude=item.latitude,
                longitude=item.longitude,
            )
            for item in ranked[:3]
        )
    else:
        candidate_infos = ()

    top_district_row = get_district(top.district_id)
    return ResolvedLocation(
        resolved_place=top.place_label,
        district_id=top.district_id,
        district_name=(
            str(top_district_row["name"]) if top_district_row is not None else top.district_name
        ),
        latitude=top.latitude,
        longitude=top.longitude,
        candidates=candidate_infos,
        source_layer=source_layer,
    )
