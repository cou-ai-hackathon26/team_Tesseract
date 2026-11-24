"""Curated crop advisory knowledge base.

This is the only component allowed to state a fact: a disease name, a symptom, a
medicine, a dose. The LLM rewrites prose around these entries and is validated
against them; it never invents one.

Scope is rice and potato only. Those are the two crops the shipped EfficientNet
checkpoint can actually name, so they are the two crops the app offers. Tomato
and chili have no class in that checkpoint, so every photo of them could only
ever be answered with "could not identify". Offering them was a dead end, not a
feature.

Every `chemical` entry carries a Bangla transliteration of the active ingredient
(`bnName`) as well as the Latin one. That is not cosmetic: the TTS model silently
deletes Latin-script spans, so a Latin-only product name would vanish from the
spoken advice while leaving a fluent, complete-sounding instruction behind.

Sources are listed per entry so the UI can cite them.
"""

from __future__ import annotations

from typing import TypedDict


class Chemical(TypedDict):
    product: str
    bnName: str
    dose: str
    note: str


class DiseaseEntry(TypedDict):
    id: str
    cropId: str
    cropName: str
    diseaseLabel: str
    bnName: str
    symptoms: list[str]
    organic: list[str]
    chemical: list[Chemical]
    prevention: list[str]
    source: str


class CropEntry(TypedDict):
    id: str
    name: str
    diseases: list[DiseaseEntry]


#: The crops this app supports. Anything else is out of scope, and the API
#: rejects it rather than storing a row it can never give sourced advice for.
SUPPORTED_CROPS: tuple[str, ...] = ("rice", "potato")


KB: list[DiseaseEntry] = [
    # ------------------------------------------------------------------ rice --
    {
        "id": "rice_blast",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "ধানের ব্লাস্ট রোগ",
        "bnName": "ব্লাস্ট",
        "symptoms": [
            "পাতায় ধূসর-সবুজ ডিম্বাকৃতি দাগ, কিনারা বাদামি",
            "দাগের মাঝখানে হালকা রঙ, পরে সেটি সাদাটে হয়ে যায়",
            "শীষের গোড়া পচে শীষ সাদা হয়ে যায়",
        ],
        "organic": [
            "আক্রান্ত পাতা ও শীষ আলাদা করে জমি থেকে সরিয়ে ফেলুন",
            "জমিতে পানি নিষ্কাশনের ব্যবস্থা রাখুন",
            "রোগমুক্ত বীজ ব্যবহার করুন",
        ],
        "chemical": [
            {
                "product": "Tricyclazole",
                "bnName": "ট্রাইসাইক্লাজল",
                "dose": "প্রতি লিটারে ০.৬ গ্রাম",
                "note": "৭ থেকে ১০ দিন পরপর সর্বোচ্চ দুইবার",
            },
            {
                "product": "Isoprothiolane",
                "bnName": "আইসোপ্রোথায়োলেন",
                "dose": "প্রতি লিটারে ১.৫ মিলি",
                "note": "শীষ বেরোনোর আগে প্রয়োগ করুন",
            },
        ],
        "prevention": [
            "অতিরিক্ত নাইট্রোজেন সার দেবেন না",
            "জাতভেদে প্রতিরোধী ধানের বীজ বপন করুন",
            "জমিতে গাছের ঘনত্ব কম রাখুন",
        ],
        "source": "BRRI - আধুনিক ধানের চাষ",
    },
    {
        "id": "rice_bacterial_blight",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "ধানের ব্যাকটেরিয়াল লিফ ব্লাইট (পাতা পোড়া রোগ)",
        "bnName": "ব্যাকটেরিয়াল ব্লাইট",
        "symptoms": [
            "পাতার ডগা ও কিনারা থেকে ঢেউ খেলানো হলুদাভ-সাদা দাগ শুকিয়ে খড়ের মতো হয়ে যায়",
            "ভোরের শিশিরে আক্রান্ত পাতায় দুধের মতো ব্যাকটেরিয়ার ফোঁটা দেখা যায়",
        ],
        "organic": [
            "জমির জমে থাকা পানি সরিয়ে ৭ থেকে ১০ দিন জমি শুকিয়ে নিন",
            "অতিরিক্ত ইউরিয়া সার দেওয়া বন্ধ করুন এবং বিঘাপ্রতি ৫ কেজি পটাশ সার উপরি প্রয়োগ করুন",
            "আক্রান্ত জমির পানি সুস্থ জমিতে যেতে দেবেন না ও কৃষি কর্মকর্তার পরামর্শ নিন",
        ],
        "chemical": [],
        "prevention": [
            "প্রতিরোধী ধানের জাত নির্বাচন করুন ও শোধিত বীজ ব্যবহার করুন",
            "সুষম মাত্রায় ইউরিয়া ও পটাশ সার প্রয়োগ করুন",
            "জমি ও আইল আগাছামুক্ত রাখুন",
        ],
        "source": "BRRI - আধুনিক ধানের চাষ",
    },
    {
        "id": "rice_brown_spot",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "ধানের বাদামি দাগ রোগ",
        "bnName": "বাদামি দাগ",
        "symptoms": [
            "পাতায় তিলের দানার মতো ডিম্বাকৃতি গাঢ় বাদামি দাগ দেখা যায়",
            "দাগের চারপাশে হালকা হলুদ আভা থাকে",
            "আক্রমণ বেশি হলে পাতা শুকিয়ে যায় এবং শীষ পুষ্ট হয় না",
        ],
        "organic": [
            "প্রথমেই জমিতে সুষম মাত্রায় পটাশ ও প্রয়োজনে দস্তা সার প্রয়োগ করে পুষ্টি ঘাটতি পূরণ করুন",
            "জমিতে রস বা হালকা পানি ধরে রাখুন এবং খরা পরিস্থিতি এড়িয়ে চলুন",
        ],
        "chemical": [
            {
                "product": "Propiconazole",
                "bnName": "প্রোপিকোনাজল",
                "dose": "প্রতি লিটারে ১ মিলি",
                "note": "পুষ্টি সংশোধনের পরও রোগ বেশি হলে প্রয়োগ করুন",
            },
            {
                "product": "Mancozeb",
                "bnName": "ম্যানকোজেব",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "প্রয়োজনে ১০ দিন পরপর",
            },
        ],
        "prevention": [
            "বপনের আগে রোগমুক্ত ও শোধিত বীজ ব্যবহার করুন",
            "সুষম সার বিশেষত পটাশ ও জিংক সার ঠিকমতো দিন",
        ],
        "source": "BRRI - আধুনিক ধানের চাষ",
    },
    {
        "id": "rice_tungro",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "ধানের টুংরো রোগ",
        "bnName": "টুংরো",
        "symptoms": [
            "গাছের পাতা আগা থেকে হলুদ ও কমলা রঙ ধারণ করে",
            "আক্রান্ত গাছের বৃদ্ধি থেমে গিয়ে খাটো হয়ে যায় ও কুশি কম হয়",
        ],
        "organic": [
            "ভাইরাস ছড়ানো বন্ধ করতে আক্রান্ত গোছা দ্রুত তুলে মাটিতে পুঁতে ফেলুন",
            "বাহক পোকা সবুজ পাতাফড়িং নিয়ন্ত্রণে আলোক ফাঁদ ও হাতজাল ব্যবহার করুন",
            "সঠিক ব্যবস্থা নিতে স্থানীয় উপ-সহকারী কৃষি কর্মকর্তার পরামর্শ নিন",
        ],
        "chemical": [],
        "prevention": [
            "রোগ প্রতিরোধী ধানের জাত চাষ করুন",
            "জমি ও আশপাশের আইল আগাছামুক্ত রাখুন",
            "আশেপাশের জমিতে একই সময়ে ধান রোপণ করুন",
        ],
        "source": "BRRI - আধুনিক ধানের চাষ",
    },
    # ---------------------------------------------------------------- potato --
    {
        "id": "potato_late_blight",
        "cropId": "potato",
        "cropName": "আলু",
        "diseaseLabel": "আলুর নাবি ধসা রোগ",
        "bnName": "নাবি ধসা",
        "symptoms": [
            "পাতার কিনারায় পানিতে ভেজা মতো বাদামি দাগ",
            "ভোরে পাতার নিচে সাদা ছত্রাকের আস্তরণ",
            "আক্রান্ত পাতা দ্রুত শুকিয়ে যাচ্ছে",
        ],
        "organic": [
            "আক্রান্ত গাছ তুলে জমি থেকে সরিয়ে ফেলুন",
            "জমিতে পানি জমতে দেবেন না",
            "ফসল আবর্তন করুন",
        ],
        "chemical": [
            {
                "product": "Mancozeb",
                "bnName": "ম্যানকোজেব",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "৭ দিন পরপর",
            },
            {
                "product": "Metalaxyl + Mancozeb",
                "bnName": "মেটালাক্সিল ও ম্যানকোজেব",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "রোগ বেশি হলে ৭ দিন পরপর",
            },
        ],
        "prevention": [
            "সুস্থ বীজ আলু ব্যবহার করুন",
            "গাছের মাঝে যথেষ্ট দূরত্ব রাখুন",
            "আগাম জাত হলে আগে রোপণ করুন",
        ],
        "source": "BARI - আলুর রোগ ব্যবস্থাপনা",
    },
    {
        "id": "potato_early_blight",
        "cropId": "potato",
        "cropName": "আলু",
        "diseaseLabel": "আলুর আগাম ধসা রোগ",
        "bnName": "আগাম ধসা",
        "symptoms": [
            "নিচের পুরনো পাতায় বাদামি কিনারাযুক্ত গোল দাগ",
            "দাগের ভেতরে একাধিক রিং দেখা যায়",
            "পাতা হলদে হয়ে শুকিয়ে ঝরে পড়ে",
        ],
        "organic": [
            "আক্রান্ত নিচের পাতা সরিয়ে ফেলুন",
            "ফসল আবর্তন করুন",
        ],
        "chemical": [
            {
                "product": "Mancozeb",
                "bnName": "ম্যানকোজেব",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "১০ দিন পরপর",
            },
            {
                "product": "Chlorothalonil",
                "bnName": "ক্লোরোথ্যালোনিল",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "১০ দিন পরপর",
            },
        ],
        "prevention": [
            "সুষম সার প্রয়োগ করুন",
            "জমিতে পানি নিষ্কাশন রাখুন",
        ],
        "source": "BARI - কৃষি প্রযুক্তি হাতবই",
    },
]


#: The checkpoint's `*_healthy` classes. They are a real result — the model
#: looked and saw no disease — so they get their own entries. They are kept out
#: of `KB` on purpose: `crops()` lists diseases, and a healthy plant is not one.
HEALTHY: dict[str, DiseaseEntry] = {
    "rice": {
        "id": "rice_healthy",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "সুস্থ ধানের গাছ",
        "bnName": "সুস্থ",
        "symptoms": [],
        "organic": [
            "এখন কোনো ধরনের রাসায়নিক বা ছত্রাকনাশক স্প্রে করবেন না",
            "সপ্তাহে একবার ক্ষেত ঘুরে পর্যবেক্ষণ করুন এবং পরিমিত পানি ও সার বজায় রাখুন",
        ],
        # No chemical for a healthy plant: naming one would invite a spray
        # nobody needs, which costs money and puts the farmer at risk.
        "chemical": [],
        "prevention": [
            "নিয়মিত জমি পর্যবেক্ষণ করুন ও আইল পরিষ্কার রাখুন",
            "সুষম মাত্রায় ইউরিয়া, টিএসপি ও পটাশ সার ব্যবহার করুন",
        ],
        "source": "BRRI - আধুনিক ধানের চাষ",
    },
    "potato": {
        "id": "potato_healthy",
        "cropId": "potato",
        "cropName": "আলু",
        "diseaseLabel": "সুস্থ আলুর গাছ",
        "bnName": "সুস্থ",
        "symptoms": [],
        "organic": [
            "এখন কোনো ধরনের ছত্রাকনাশক বা রাসায়নিক স্প্রে করবেন না",
            "নিয়মিত ৩ থেকে ৪ দিন পরপর ক্ষেত ঘুরে দেখুন ও নালার পানি নিষ্কাশন ঠিক রাখুন",
        ],
        "chemical": [],
        "prevention": [
            "কুয়াশাচ্ছন্ন আবহাওয়ায় পাতার নিচে নিয়মিত নজর রাখুন",
            "সুষম সার প্রয়োগ করুন এবং সময়মতো গাছের গোড়ায় মাটি তুলে দিন",
        ],
        "source": "BARI - কৃষি প্রযুক্তি হাতবই",
    },
}

_DISEASES = {entry["id"]: entry for entry in KB}
_HEALTHY_BY_ID = {entry["id"]: entry for entry in HEALTHY.values()}
_DISEASE_ALIASES = {
    "rice_bacterial_leaf_blight": "rice_bacterial_blight",
}


def get_disease(disease_id: str) -> DiseaseEntry | None:
    """Look up an entry. Healthy and alias entries resolve here too."""
    canonical = _DISEASE_ALIASES.get(disease_id, disease_id)
    return _DISEASES.get(canonical) or _HEALTHY_BY_ID.get(canonical)


def get_healthy(crop_id: str) -> DiseaseEntry | None:
    return HEALTHY.get(crop_id)


def diseases_for_crop(crop_id: str) -> list[DiseaseEntry]:
    return [entry for entry in KB if entry["cropId"] == crop_id]


def crops() -> list[CropEntry]:
    seen: list[CropEntry] = []
    for entry in KB:
        if not any(crop["id"] == entry["cropId"] for crop in seen):
            seen.append(
                {
                    "id": entry["cropId"],
                    "name": entry["cropName"],
                    "diseases": diseases_for_crop(entry["cropId"]),
                }
            )
    return seen
