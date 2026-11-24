"""The 64 districts of Bangladesh, with coordinates for the weather lookup.

Districts, not a geolocation prompt: one tap, no permission dialog, works
indoors, and coarse enough that the location never becomes identifying.
"""

from __future__ import annotations

import math

DISTRICTS: list[dict[str, object]] = [
    # Dhaka division
    {"id": "dhaka", "name": "ঢাকা", "division": "ঢাকা", "latitude": 23.8103, "longitude": 90.4125},
    {"id": "gazipur", "name": "গাজীপুর", "division": "ঢাকা", "latitude": 23.9999, "longitude": 90.4203},
    {
        "id": "manikganj",
        "name": "মানিকগঞ্জ",
        "division": "ঢাকা",
        "latitude": 23.8617,
        "longitude": 90.0003,
    },
    {
        "id": "munshiganj",
        "name": "মুন্সিগঞ্জ",
        "division": "ঢাকা",
        "latitude": 23.5422,
        "longitude": 90.5305,
    },
    {
        "id": "narayanganj",
        "name": "নারায়ণগঞ্জ",
        "division": "ঢাকা",
        "latitude": 23.6238,
        "longitude": 90.4998,
    },
    {
        "id": "narsingdi",
        "name": "নরসিংদী",
        "division": "ঢাকা",
        "latitude": 23.9322,
        "longitude": 90.7151,
    },
    {"id": "tangail", "name": "টাঙ্গাইল", "division": "ঢাকা", "latitude": 24.2513, "longitude": 89.9167},
    {
        "id": "kishoreganj",
        "name": "কিশোরগঞ্জ",
        "division": "ঢাকা",
        "latitude": 24.4449,
        "longitude": 90.7766,
    },
    {
        "id": "faridpur",
        "name": "ফরিদপুর",
        "division": "ঢাকা",
        "latitude": 23.6070,
        "longitude": 89.8429,
    },
    {
        "id": "gopalganj",
        "name": "গোপালগঞ্জ",
        "division": "ঢাকা",
        "latitude": 23.0050,
        "longitude": 89.8266,
    },
    {
        "id": "madaripur",
        "name": "মাদারীপুর",
        "division": "ঢাকা",
        "latitude": 23.1640,
        "longitude": 90.1897,
    },
    {"id": "rajbari", "name": "রাজবাড়ী", "division": "ঢাকা", "latitude": 23.7574, "longitude": 89.6445},
    {
        "id": "shariatpur",
        "name": "শরীয়তপুর",
        "division": "ঢাকা",
        "latitude": 23.2423,
        "longitude": 90.4348,
    },
    # Chattogram division
    {
        "id": "chattogram",
        "name": "চট্টগ্রাম",
        "division": "চট্টগ্রাম",
        "latitude": 22.3569,
        "longitude": 91.7832,
    },
    {
        "id": "coxsbazar",
        "name": "কক্সবাজার",
        "division": "চট্টগ্রাম",
        "latitude": 21.4272,
        "longitude": 92.0058,
    },
    {
        "id": "comilla",
        "name": "কুমিল্লা",
        "division": "চট্টগ্রাম",
        "latitude": 23.4607,
        "longitude": 91.1809,
    },
    {
        "id": "brahmanbaria",
        "name": "ব্রাহ্মণবাড়িয়া",
        "division": "চট্টগ্রাম",
        "latitude": 23.9571,
        "longitude": 91.1119,
    },
    {
        "id": "chandpur",
        "name": "চাঁদপুর",
        "division": "চট্টগ্রাম",
        "latitude": 23.2333,
        "longitude": 90.6712,
    },
    {"id": "feni", "name": "ফেনী", "division": "চট্টগ্রাম", "latitude": 23.0159, "longitude": 91.3976},
    {
        "id": "lakshmipur",
        "name": "লক্ষ্মীপুর",
        "division": "চট্টগ্রাম",
        "latitude": 22.9447,
        "longitude": 90.8281,
    },
    {
        "id": "noakhali",
        "name": "নোয়াখালী",
        "division": "চট্টগ্রাম",
        "latitude": 22.8696,
        "longitude": 91.0995,
    },
    {
        "id": "khagrachhari",
        "name": "খাগড়াছড়ি",
        "division": "চট্টগ্রাম",
        "latitude": 23.1193,
        "longitude": 91.9847,
    },
    {
        "id": "rangamati",
        "name": "রাঙ্গামাটি",
        "division": "চট্টগ্রাম",
        "latitude": 22.6533,
        "longitude": 92.1735,
    },
    {
        "id": "bandarban",
        "name": "বান্দরবান",
        "division": "চট্টগ্রাম",
        "latitude": 22.1953,
        "longitude": 92.2184,
    },
    # Rajshahi division
    {
        "id": "rajshahi",
        "name": "রাজশাহী",
        "division": "রাজশাহী",
        "latitude": 24.3745,
        "longitude": 88.6042,
    },
    {"id": "bogura", "name": "বগুড়া", "division": "রাজশাহী", "latitude": 24.8465, "longitude": 89.3774},
    {"id": "pabna", "name": "পাবনা", "division": "রাজশাহী", "latitude": 24.0064, "longitude": 89.2372},
    {
        "id": "sirajganj",
        "name": "সিরাজগঞ্জ",
        "division": "রাজশাহী",
        "latitude": 24.4534,
        "longitude": 89.7007,
    },
    {
        "id": "natore",
        "name": "নাটোর",
        "division": "রাজশাহী",
        "latitude": 24.4206,
        "longitude": 89.0003,
    },
    {
        "id": "naogaon",
        "name": "নওগাঁ",
        "division": "রাজশাহী",
        "latitude": 24.7936,
        "longitude": 88.9318,
    },
    {
        "id": "chapainawabganj",
        "name": "চাঁপাইনবাবগঞ্জ",
        "division": "রাজশাহী",
        "latitude": 24.5965,
        "longitude": 88.2775,
    },
    {
        "id": "joypurhat",
        "name": "জয়পুরহাট",
        "division": "রাজশাহী",
        "latitude": 25.0947,
        "longitude": 89.0227,
    },
    # Khulna division
    {"id": "khulna", "name": "খুলনা", "division": "খুলনা", "latitude": 22.8456, "longitude": 89.5403},
    {"id": "jashore", "name": "যশোর", "division": "খুলনা", "latitude": 23.1664, "longitude": 89.2081},
    {
        "id": "satkhira",
        "name": "সাতক্ষীরা",
        "division": "খুলনা",
        "latitude": 22.7185,
        "longitude": 89.0705,
    },
    {
        "id": "bagerhat",
        "name": "বাগেরহাট",
        "division": "খুলনা",
        "latitude": 22.6516,
        "longitude": 89.7859,
    },
    {
        "id": "kushtia",
        "name": "কুষ্টিয়া",
        "division": "খুলনা",
        "latitude": 23.9013,
        "longitude": 89.1206,
    },
    {
        "id": "chuadanga",
        "name": "চুয়াডাঙ্গা",
        "division": "খুলনা",
        "latitude": 23.6402,
        "longitude": 88.8418,
    },
    {
        "id": "jhenaidah",
        "name": "ঝিনাইদহ",
        "division": "খুলনা",
        "latitude": 23.5449,
        "longitude": 89.1539,
    },
    {"id": "magura", "name": "মাগুরা", "division": "খুলনা", "latitude": 23.4855, "longitude": 89.4198},
    {
        "id": "meherpur",
        "name": "মেহেরপুর",
        "division": "খুলনা",
        "latitude": 23.7622,
        "longitude": 88.6310,
    },
    {"id": "narail", "name": "নড়াইল", "division": "খুলনা", "latitude": 23.1725, "longitude": 89.5127},
    # Barishal division
    {
        "id": "barishal",
        "name": "বরিশাল",
        "division": "বরিশাল",
        "latitude": 22.7010,
        "longitude": 90.3535,
    },
    {"id": "bhola", "name": "ভোলা", "division": "বরিশাল", "latitude": 22.6859, "longitude": 90.6482},
    {
        "id": "barguna",
        "name": "বরগুনা",
        "division": "বরিশাল",
        "latitude": 22.1596,
        "longitude": 90.1263,
    },
    {
        "id": "patuakhali",
        "name": "পটুয়াখালী",
        "division": "বরিশাল",
        "latitude": 22.3596,
        "longitude": 90.3298,
    },
    {
        "id": "pirojpur",
        "name": "পিরোজপুর",
        "division": "বরিশাল",
        "latitude": 22.5841,
        "longitude": 89.9720,
    },
    {
        "id": "jhalokathi",
        "name": "ঝালকাঠি",
        "division": "বরিশাল",
        "latitude": 22.6406,
        "longitude": 90.1987,
    },
    # Sylhet division
    {
        "id": "sylhet",
        "name": "সিলেট",
        "division": "সিলেট",
        "latitude": 24.8949,
        "longitude": 91.8687,
    },
    {
        "id": "moulvibazar",
        "name": "মৌলভীবাজার",
        "division": "সিলেট",
        "latitude": 24.4829,
        "longitude": 91.7774,
    },
    {
        "id": "habiganj",
        "name": "হবিগঞ্জ",
        "division": "সিলেট",
        "latitude": 24.3745,
        "longitude": 91.4155,
    },
    {
        "id": "sunamganj",
        "name": "সুনামগঞ্জ",
        "division": "সিলেট",
        "latitude": 25.0658,
        "longitude": 91.3950,
    },
    # Rangpur division
    {
        "id": "rangpur",
        "name": "রংপুর",
        "division": "রংপুর",
        "latitude": 25.7439,
        "longitude": 89.2752,
    },
    {
        "id": "dinajpur",
        "name": "দিনাজপুর",
        "division": "রংপুর",
        "latitude": 25.6217,
        "longitude": 88.6350,
    },
    {
        "id": "nilphamari",
        "name": "নীলফামারী",
        "division": "রংপুর",
        "latitude": 25.9313,
        "longitude": 88.8560,
    },
    {
        "id": "kurigram",
        "name": "কুড়িগ্রাম",
        "division": "রংপুর",
        "latitude": 25.8072,
        "longitude": 89.6295,
    },
    {
        "id": "gaibandha",
        "name": "গাইবান্ধা",
        "division": "রংপুর",
        "latitude": 25.3286,
        "longitude": 89.5281,
    },
    {
        "id": "lalmonirhat",
        "name": "লালমনিরহাট",
        "division": "রংপুর",
        "latitude": 25.9923,
        "longitude": 89.2847,
    },
    {
        "id": "thakurgaon",
        "name": "ঠাকুরগাঁও",
        "division": "রংপুর",
        "latitude": 26.0337,
        "longitude": 88.4619,
    },
    {
        "id": "panchagarh",
        "name": "পঞ্চগড়",
        "division": "রংপুর",
        "latitude": 26.3411,
        "longitude": 88.5542,
    },
    # Mymensingh division
    {
        "id": "mymensingh",
        "name": "ময়মনসিংহ",
        "division": "ময়মনসিংহ",
        "latitude": 24.7471,
        "longitude": 90.4203,
    },
    {
        "id": "jamalpur",
        "name": "জামালপুর",
        "division": "ময়মনসিংহ",
        "latitude": 24.9375,
        "longitude": 89.9372,
    },
    {
        "id": "netrokona",
        "name": "নেত্রকোনা",
        "division": "ময়মনসিংহ",
        "latitude": 24.8103,
        "longitude": 90.7279,
    },
    {
        "id": "sherpur",
        "name": "শেরপুর",
        "division": "ময়মনসিংহ",
        "latitude": 25.0205,
        "longitude": 90.0153,
    },
]

_BY_ID = {str(d["id"]): d for d in DISTRICTS}


def get_district(district_id: str) -> dict[str, object] | None:
    return _BY_ID.get(district_id)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * r_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest_district(latitude: float, longitude: float) -> dict[str, object]:
    """Return the Bangladesh district closest to the provided coordinates."""
    return min(
        DISTRICTS,
        key=lambda item: _haversine_km(
            latitude,
            longitude,
            float(str(item["latitude"])),
            float(str(item["longitude"])),
        ),
    )


def default_district() -> dict[str, object]:
    return _BY_ID["dhaka"]
