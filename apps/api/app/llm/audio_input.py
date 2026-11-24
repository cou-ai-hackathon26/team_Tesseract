"""Bangla voice & text question extraction powered by Gemini Flash + offline fallback.

Failure here never blocks leaf disease analysis: an unconfigured key, timeout,
or inaudible clip logs a warning and degrades cleanly to deterministic offline
keyword/regex extraction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..grounding.districts import DISTRICTS
from .writer import (
    _RETRYABLE_GEMINI_CODES,
    DEFAULT_GEMINI_MODEL,
    FALLBACK_GEMINI_MODELS,
    normalise,
)

log = logging.getLogger(__name__)

SpokenCrop = Literal["rice", "potato", "other", "none"]

_MARKDOWN_NOISE = re.compile(r"[*_`#>\[\]{}|]")

_TRANSCRIPTION_PROMPT = (
    "এই অডিও ক্লিপে বাংলাদেশের একজন কৃষকের ফসল বা রোগ সংক্রান্ত বাংলা প্রশ্ন রয়েছে। "
    "কৃষক যা বলেছেন, কেবল সেই প্রশ্নটি সঠিক ও সংক্ষিপ্ত বাংলা অক্ষরে লিখুন। "
    "কোনো উত্তর দেবেন না, কোনো ভূমিকা, মার্কডাউন বা অতিরিক্ত মন্তব্য লিখবেন না। "
    "যদি কথা অস্পষ্ট বা নীরব হয়, তবে সম্পূর্ণ খালি উত্তর দিন।"
)

_EXTRACTION_PROMPT = (
    "You extract structured agricultural context from a Bangladeshi farmer's "
    "Bangla voice/text question.\n"
    "Return JSON only with these exact fields:\n"
    "- transcript_bn: Exact clean Bangla transcript of the spoken audio "
    "(null if no audio or inaudible).\n"
    "- spoken_place_bn: Village, union, upazila, bazaar, or area name in Bangla if mentioned "
    '(e.g. "বুড়িচং", "শেরপুর"), else null.\n'
    "- spoken_place_en: Clean English transliteration of spoken_place_bn "
    '(e.g. "Burichang", "Sherpur"), else null.\n'
    "- spoken_district_en: Parent Bangladesh district in English ONLY if a parent district "
    'was explicitly spoken (e.g. "Cumilla", "Bogura"), else null.\n'
    "- spoken_plot_hint_bn: Field/plot condition hint in Bangla if mentioned "
    '(e.g. "নিচু জমি", "পুকুরের পাশের জমি", "জলাবদ্ধ জমি"), else null.\n'
    '- spoken_crop: One of "rice", "potato", "other", or "none" based on the crop mentioned.\n'
    '- spoken_crop_bn: Crop name in Bangla if mentioned (e.g. "ধান", "আলু", "টমেটো"), else null.\n'
    "Do not invent locations or crops that were not mentioned."
)


class QuestionContext(BaseModel):
    """Structured fields extracted from farmer voice audio and/or text question."""

    transcript_bn: str | None = None
    spoken_place_bn: str | None = None
    spoken_place_en: str | None = None
    spoken_district_en: str | None = None
    spoken_plot_hint_bn: str | None = None
    spoken_crop: SpokenCrop = "none"
    spoken_crop_bn: str | None = None


_DISTRICT_CANONICAL_EN: dict[str, str] = {
    "comilla": "Cumilla",
    "coxsbazar": "Cox's Bazar",
    "chapainawabganj": "Chapainawabganj",
    "bogura": "Bogura",
    "chattogram": "Chattogram",
    "jashore": "Jashore",
    "barishal": "Barishal",
    "brahmanbaria": "Brahmanbaria",
    "khagrachhari": "Khagrachhari",
    "moulvibazar": "Moulvibazar",
    "lalmonirhat": "Lalmonirhat",
    "thakurgaon": "Thakurgaon",
    "panchagarh": "Panchagarh",
    "narayanganj": "Narayanganj",
    "munshiganj": "Munshiganj",
    "kishoreganj": "Kishoreganj",
    "shariatpur": "Shariatpur",
    "jhenaidah": "Jhenaidah",
    "chuadanga": "Chuadanga",
    "patuakhali": "Patuakhali",
    "jhalokathi": "Jhalokathi",
    "sunamganj": "Sunamganj",
    "gaibandha": "Gaibandha",
    "nilphamari": "Nilphamari",
    "mymensingh": "Mymensingh",
}

_DISTRICT_ALIAS_EN: dict[str, tuple[str, str]] = {
    "comilla": ("কুমিল্লা", "Cumilla"),
    "cumilla": ("কুমিল্লা", "Cumilla"),
    "bogra": ("বগুড়া", "Bogura"),
    "bogura": ("বগুড়া", "Bogura"),
    "chittagong": ("চট্টগ্রাম", "Chattogram"),
    "chattogram": ("চট্টগ্রাম", "Chattogram"),
    "jessore": ("যশোর", "Jashore"),
    "jashore": ("যশোর", "Jashore"),
    "barisal": ("বরিশাল", "Barishal"),
    "barishal": ("বরিশাল", "Barishal"),
    "cox's bazar": ("কক্সবাজার", "Cox's Bazar"),
    "coxs bazar": ("কক্সবাজার", "Cox's Bazar"),
    "coxsbazar": ("কক্সবাজার", "Cox's Bazar"),
    "nawabganj": ("চাঁপাইনবাবগঞ্জ", "Chapainawabganj"),
    "chapainawabganj": ("চাঁপাইনবাবগঞ্জ", "Chapainawabganj"),
    "netrakona": ("নেত্রকোনা", "Netrokona"),
    "netrokona": ("নেত্রকোনা", "Netrokona"),
    "khagrachari": ("খাগড়াছড়ি", "Khagrachhari"),
    "jhalokati": ("ঝালকাঠি", "Jhalokathi"),
}


def _canonical_district_en(district_id: str) -> str:
    return _DISTRICT_CANONICAL_EN.get(district_id, district_id.capitalize())


_DISTRICT_BY_BN: dict[str, str] = {}
for _d in DISTRICTS:
    _did = str(_d["id"])
    _dbn = normalise(str(_d["name"]))
    _den = _canonical_district_en(_did)
    _DISTRICT_BY_BN[_dbn] = _den
    _DISTRICT_ALIAS_EN.setdefault(_did, (_dbn, _den))
    _DISTRICT_ALIAS_EN.setdefault(_den.lower(), (_dbn, _den))

#: Rural upazilas, unions, bazaars, and known duplicate Bangladesh place names.
#: Duplicate names (e.g. Sherpur, Kaliganj, Shibganj, Pirganj, Sreepur) keep
#: parent district as None so the geocoder disambiguates via spoken district / GPS.
_KNOWN_PLACES_BN: dict[str, str] = {
    "বুড়িচং": "Burichang",
    "দেবিদ্বার": "Debidwar",
    "দাউদকান্দি": "Daudkandi",
    "চৌদ্দগ্রাম": "Chauddagram",
    "লাকসাম": "Laksam",
    "মুরাদনগর": "Muradnagar",
    "বরুড়া": "Barura",
    "চান্দিনা": "Chandina",
    "হোমনা": "Homna",
    "তিতাস": "Titas",
    "মনোহরগঞ্জ": "Manoharganj",
    "নাঙ্গলকোট": "Nangalkot",
    "মেঘনা": "Meghna",
    "সদর দক্ষিণ": "Sadar Dakshin",
    "শেরপুর": "Sherpur",
    "কালীগঞ্জ": "Kaliganj",
    "কালিগঞ্জ": "Kaliganj",
    "শিবগঞ্জ": "Shibganj",
    "সোনাতলা": "Sonatola",
    "গাবতলী": "Gabtali",
    "ধুনট": "Dhunat",
    "কাহালু": "Kahaloo",
    "দুপচাঁচিয়া": "Dupchanchia",
    "আদমদীঘি": "Adamdighi",
    "নন্দীগ্রাম": "Nandigram",
    "সারিয়াকান্দি": "Sariakandi",
    "শাহজাহানপুর": "Shajahanpur",
    "কোম্পানীগঞ্জ": "Companiganj",
    "সাভার": "Savar",
    "ধামরাই": "Dhamrai",
    "কেরানীগঞ্জ": "Keraniganj",
    "শ্রীপুর": "Sreepur",
    "কালিয়াকৈর": "Kaliakair",
    "কাপাসিয়া": "Kapasia",
    "পীরগঞ্জ": "Pirganj",
    "গোবিন্দগঞ্জ": "Gobindaganj",
    "বদরগঞ্জ": "Badarganj",
    "মিঠাপুকুর": "Mithapukur",
    "সৈয়দপুর": "Saidpur",
    "ত্রিশাল": "Trishal",
    "ভালুকা": "Bhaluka",
    "মুক্তাগাছা": "Muktagacha",
    "ফুলপুর": "Phulpur",
    "গফরগাঁও": "Gaffargaon",
    "মধুপুর": "Madhupur",
    "সখীপুর": "Sakhipur",
    "মির্জাপুর": "Mirzapur",
    "সোনারগাঁও": "Sonargaon",
    "সোনারগাঁ": "Sonargaon",
    "আড়াইহাজার": "Araihazar",
    "রূপগঞ্জ": "Rupganj",
    "শ্রীমঙ্গল": "Sreemangal",
    "মাধবপুর": "Madhabpur",
    "জকিগঞ্জ": "Zakiganj",
    "রামগঞ্জ": "Ramganj",
    "সোনাইমুড়ী": "Sonaimuri",
    "বেগমগঞ্জ": "Begumganj",
    "ফটিকছড়ি": "Fatikchhari",
    "সীতাকুণ্ড": "Sitakunda",
    "মিরসরাই": "Mirsharai",
    "পটিয়া": "Patiya",
    "বাঁশখালী": "Banshkhali",
    "চকরিয়া": "Chakaria",
    "চকোরিয়া": "Chakaria",
    "টেকনাফ": "Teknaf",
    "চাটমোহর": "Chatmohar",
    "ঈশ্বরদী": "Ishwardi",
    "গোদাগাড়ী": "Godagari",
    "তানোর": "Tanore",
    "মোংলা": "Mongla",
    "কোটালীপাড়া": "Kotalipara",
    "টুঙ্গিপাড়া": "Tungipara",
    "নালিতাবাড়ী": "Nalitabari",
    "নকলা": "Nakla",
    "ঝিনাইগাতী": "Jhenaigati",
    "শ্রীবরদী": "Sreebardi",
}

_KNOWN_PLACES_EN: dict[str, tuple[str, str]] = {
    en.lower(): (bn, en) for bn, en in _KNOWN_PLACES_BN.items()
}

_CROP_STOPWORDS = frozenset(
    {
        "ধান",
        "আলু",
        "টমেটো",
        "মরিচ",
        "বেগুন",
        "গম",
        "ভুট্টা",
        "সরিষা",
        "পাট",
        "আম",
        "শসা",
        "লাউ",
        "পেঁয়াজ",
        "কলা",
        "সবজি",
    }
)

_PLACE_STOPWORDS = frozenset(
    {
        "আমার",
        "আমাদের",
        "তোমার",
        "তোমাদের",
        "আপনার",
        "আপনাদের",
        "এই",
        "ওই",
        "সেই",
        "নিজ",
        "নিজের",
        "নিচু",
        "উঁচু",
        "পাশের",
        "পাশে",
        "জমি",
        "জমির",
        "জমিতে",
        "ক্ষেত",
        "ক্ষেতের",
        "ক্ষেতে",
        "ফসল",
        "ফসলের",
        "গাছ",
        "গাছের",
        "পাতা",
        "পাতায়",
        "পাতার",
        "স্থানীয়",
        "কোন",
        "কোথায়",
        "এখানে",
        "ওখানে",
        "সেখানে",
        "এলাকা",
        "এলাকার",
        "গ্রাম",
        "গ্রামের",
        "বাজার",
        "বাজারের",
        "উপজেলা",
        "উপজেলার",
        "ইউনিয়ন",
        "ইউনিয়নের",
        "থানা",
        "থানার",
        "জেলা",
        "জেলার",
        "পুকুর",
        "পুকুরের",
        "নদী",
        "নদীর",
        "খাল",
        "খালের",
        "বিল",
        "বিলের",
        "হাওর",
        "হাওরের",
        "পানি",
        "বৃষ্টি",
        "ছবি",
        "প্রশ্ন",
    }
    | _CROP_STOPWORDS
)

_BN_PLACE_SUFFIX_RE = re.compile(
    r"(?<![\u0980-\u09FF])"
    r"([\u0980-\u09FF]{2,20})(?:ের|র|ে)?"
    r"\s+(?:গ্রাম(?:ের|ে)?|ইউনিয়ন(?:ের|ে)?|উপজেলা(?:য়|র)?|বাজার(?:ের|ে)?|থানা(?:য়|র)?|এলাকা(?:য়|র)?|হাট(?:ের|ে)?|পাড়া(?:য়|র)?)"
    r"(?![\u0980-\u09FF])"
)
_BN_FROM_PLACE_RE = re.compile(
    r"(?<![\u0980-\u09FF])([\u0980-\u09FF]{2,20})(?:ের|র|ে|য়|য়ে|তে)?\s+থেকে\s+(?:বলছি|লিখছি|জানতে|পাঠাচ্ছি)"
)

_PLOT_HINT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"পুকুর(?:ের)?\s*পা(?:শে|ড়ে)র?\s*জমি(?:তে|র)?"), "পুকুরের পাশের জমি"),
    (re.compile(r"খাল(?:ের)?\s*পা(?:শে|ড়ে)র?\s*জমি(?:তে|র)?"), "খালের পাশের জমি"),
    (re.compile(r"নদী(?:র)?\s*পা(?:শে|ড়ে)র?\s*জমি(?:তে|র)?"), "নদীর পাড়ের জমি"),
    (re.compile(r"জলাবদ্ধ\s*(?:জমি|ক্ষেত)(?:তে|র)?"), "জলাবদ্ধ জমি"),
    (re.compile(r"পানি\s*জম(?:া|ে\s*থাকা)\s*(?:জমি|ক্ষেত)(?:তে|র)?"), "পানি জমে থাকা জমি"),
    (re.compile(r"নিচু\s*(?:জমি|ক্ষেত|এলাকা)(?:তে|র|য়)?"), "নিচু জমি"),
    (re.compile(r"উঁচু\s*(?:জমি|ক্ষেত)(?:তে|র)?"), "উঁচু জমি"),
    (re.compile(r"বিল(?:ের)?\s*(?:জমি|ক্ষেত)(?:তে|র)?"), "বিলের জমি"),
    (re.compile(r"হাওর(?:ের)?\s*(?:জমি|ক্ষেত)(?:তে|র)?"), "হাওরের জমি"),
    (re.compile(r"চর(?:ের)?\s*(?:জমি|ক্ষেত)(?:তে|র)?"), "চরের জমি"),
    (re.compile(r"ছায়া(?:যুক্ত|র)?\s*(?:জমি|জায়গা|ক্ষেত)(?:তে|র)?"), "ছায়াযুক্ত জমি"),
)

_CROP_PATTERNS: tuple[tuple[re.Pattern[str], SpokenCrop, str], ...] = (
    (
        re.compile(
            r"(?<![\u0980-\u09FF])(?:ধান(?:ের|ে|ক্ষেত|গাছ|গাছে|গাছের)?|বোরো|আমন|আউশ)(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])(?:rice|paddy)(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "rice",
        "ধান",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])আলু(?:র|তে|ক্ষেত|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])potato(?:es)?(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "potato",
        "আলু",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])টমেটো(?:র|তে|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])tomato(?:es)?(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "টমেটো",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])মরিচ(?:ের|ে|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])chill?i(?:es)?(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "মরিচ",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])বেগুন(?:ের|ে|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])(?:brinjal|eggplant)(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "বেগুন",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])গম(?:ের|ে|ক্ষেত|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])wheat(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "গম",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])ভুট্টা(?:র|য়|ক্ষেত|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])(?:maize|corn)(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "ভুট্টা",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])সরিষা(?:র|য়|ক্ষেত|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])mustard(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "সরিষা",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])পাট(?:ের|ে|ক্ষেত|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])jute(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "পাট",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])আম(?:ের|গাছ|গাছে|গাছের|বাগান|বাগানে)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])mango(?:es)?(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "আম",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])শসা(?:র|য়|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])cucumber(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "শসা",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])লাউ(?:য়ের|য়ের|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])gourd(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "লাউ",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])পেঁয়াজ(?:ের|ে|ক্ষেত|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])onion(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "পেঁয়াজ",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])কলা(?:র|গাছ|গাছে|গাছের|বাগান)?(?![\u0980-\u09FF])"
            r"|(?<![A-Za-z])banana(?![A-Za-z])",
            re.IGNORECASE,
        ),
        "other",
        "কলা",
    ),
    (
        re.compile(
            r"(?<![\u0980-\u09FF])(?:মিষ্টিকুমড়া|কুমড়া|পটল|বাঁধাকপি|ফুলকপি|সবজি)(?:র|তে|য়|ের|গাছ|গাছে|গাছের)?(?![\u0980-\u09FF])"
        ),
        "other",
        "সবজি",
    ),
)

_BN_TRANSLIT: dict[str, str] = {
    "অ": "a",
    "আ": "a",
    "ই": "i",
    "ঈ": "i",
    "উ": "u",
    "ঊ": "u",
    "ঋ": "ri",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
    "া": "a",
    "ি": "i",
    "ী": "i",
    "ু": "u",
    "ূ": "u",
    "ৃ": "ri",
    "ে": "e",
    "ৈ": "oi",
    "ো": "o",
    "ৌ": "ou",
    "ক": "k",
    "খ": "kh",
    "গ": "g",
    "ঘ": "gh",
    "ঙ": "ng",
    "চ": "ch",
    "ছ": "chh",
    "জ": "j",
    "ঝ": "jh",
    "ঞ": "n",
    "ট": "t",
    "ঠ": "th",
    "ড": "d",
    "ঢ": "dh",
    "ণ": "n",
    "ত": "t",
    "থ": "th",
    "দ": "d",
    "ধ": "dh",
    "ন": "n",
    "প": "p",
    "ফ": "f",
    "ব": "b",
    "ভ": "bh",
    "ম": "m",
    "য": "j",
    "র": "r",
    "ল": "l",
    "শ": "sh",
    "ষ": "sh",
    "স": "s",
    "হ": "h",
    "ড়": "r",
    "ঢ়": "rh",
    "য়": "y",
    "ৎ": "t",
    "ং": "ng",
    "ঃ": "h",
    "ঁ": "n",
    "্": "",
}


def _transliterate_bn_to_en(word_bn: str) -> str:
    if word_bn in _KNOWN_PLACES_BN:
        return _KNOWN_PLACES_BN[word_bn]
    if word_bn in _DISTRICT_BY_BN:
        return _DISTRICT_BY_BN[word_bn]
    out: list[str] = []
    for ch in normalise(word_bn):
        mapped = _BN_TRANSLIT.get(ch)
        if mapped is not None:
            out.append(mapped)
        elif ch.isascii() and ch.isalnum():
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
    joined = "".join(out).strip()
    return joined.title() if joined else ""


def _candidate_models(settings: Settings) -> list[str]:
    primary = settings.gemini_model.strip() or DEFAULT_GEMINI_MODEL
    candidates = [primary]
    for model in FALLBACK_GEMINI_MODELS:
        if model not in candidates:
            candidates.append(model)
    return candidates


def _build_client(api_key: str, timeout_ms: int) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )


def _clean_transcript(raw_text: str, max_chars: int) -> str | None:
    cleaned = _MARKDOWN_NOISE.sub("", raw_text)
    cleaned = normalise(cleaned).strip("\"'“”‘’ ")
    if not cleaned:
        return None
    return cleaned[:max_chars].strip() or None


def _clean_opt(value: str | None, max_chars: int = 120) -> str | None:
    if not value:
        return None
    cleaned = _MARKDOWN_NOISE.sub("", str(value))
    cleaned = normalise(cleaned).strip("\"'“”‘’ ")
    if not cleaned or cleaned.lower() in {"null", "none", "n/a"}:
        return None
    return cleaned[:max_chars].strip() or None


def _strip_bn_case_suffix(token: str) -> str:
    for suffix in ("ের", "য়ের", "য়ে", "তে", "র", "য়", "ে"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            stem = token[: -len(suffix)]
            if stem in _KNOWN_PLACES_BN or stem in _DISTRICT_BY_BN:
                return stem
    return token


def _match_bn_word(text: str, term: str) -> re.Match[str] | None:
    pattern = re.compile(
        rf"(?<![\u0980-\u09FF]){re.escape(term)}(?:ের|য়ের|য়ে|তে|র|য়|ে)?(?![\u0980-\u09FF])"
    )
    return pattern.search(text)


def _match_en_word(text: str, term: str) -> re.Match[str] | None:
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", re.IGNORECASE)
    return pattern.search(text)


def extract_question_context_offline(
    text: str | None,
    *,
    transcript_bn: str | None = None,
) -> QuestionContext:
    """Deterministically extract crop, location, and plot hints without network calls."""
    cleaned_transcript = _clean_opt(transcript_bn, 300)
    combined = normalise(text or cleaned_transcript or "")
    if not combined:
        return QuestionContext(transcript_bn=cleaned_transcript)

    # 1. Crop extraction: earliest explicit crop mention wins.
    best_crop: tuple[int, SpokenCrop, str] | None = None
    for pattern, crop_code, crop_bn in _CROP_PATTERNS:
        match = pattern.search(combined)
        if match and (best_crop is None or match.start() < best_crop[0]):
            best_crop = (match.start(), crop_code, crop_bn)

    spoken_crop: SpokenCrop = best_crop[1] if best_crop else "none"
    spoken_crop_bn: str | None = best_crop[2] if best_crop else None

    # 2. Plot hint extraction.
    spoken_plot_hint_bn: str | None = None
    for pattern, canonical_hint in _PLOT_HINT_PATTERNS:
        if pattern.search(combined):
            spoken_plot_hint_bn = canonical_hint
            break

    # 3. Parent district + village/upazila/bazaar extraction.
    # Known duplicate district/place names (like Sherpur) do NOT force a parent
    # district unless "শেরপুর জেলা" was explicitly spoken or another district appears.
    explicit_sherpur_district = bool(
        re.search(r"শেরপুর\s+জেলা(?:য়|র|তে)?(?![\u0980-\u09FF])", combined)
    )
    matched_districts: list[tuple[int, str, str]] = []
    for bn_name, en_name in _DISTRICT_BY_BN.items():
        if bn_name == "শেরপুর" and not explicit_sherpur_district:
            continue
        match = _match_bn_word(combined, bn_name)
        if match:
            matched_districts.append((match.start(), bn_name, en_name))

    for en_alias, (bn_name, en_name) in _DISTRICT_ALIAS_EN.items():
        if en_alias == "sherpur" and not re.search(r"\bsherpur\s+district\b", combined, re.I):
            continue
        match = _match_en_word(combined, en_alias)
        if match:
            matched_districts.append((match.start(), bn_name, en_name))

    matched_districts.sort(key=lambda item: item[0])
    spoken_district_en = matched_districts[0][2] if matched_districts else None

    matched_places: list[tuple[int, str, str]] = []
    for bn_place, en_place in _KNOWN_PLACES_BN.items():
        match = _match_bn_word(combined, bn_place)
        if match:
            matched_places.append((match.start(), bn_place, en_place))

    for en_alias, (bn_place, en_place) in _KNOWN_PLACES_EN.items():
        match = _match_en_word(combined, en_alias)
        if match:
            matched_places.append((match.start(), bn_place, en_place))

    if not matched_places:
        for pattern in (_BN_PLACE_SUFFIX_RE, _BN_FROM_PLACE_RE):
            for match in pattern.finditer(combined):
                candidate_bn = _strip_bn_case_suffix(normalise(match.group(1)))
                if (
                    candidate_bn
                    and candidate_bn not in _PLACE_STOPWORDS
                    and candidate_bn not in _DISTRICT_BY_BN
                ):
                    matched_places.append(
                        (match.start(), candidate_bn, _transliterate_bn_to_en(candidate_bn))
                    )
                    break
            if matched_places:
                break

    matched_places.sort(key=lambda item: item[0])

    spoken_place_bn: str | None = None
    spoken_place_en: str | None = None
    if matched_places:
        _, spoken_place_bn, spoken_place_en = matched_places[0]
    elif matched_districts:
        _, spoken_place_bn, spoken_place_en = matched_districts[0]

    return QuestionContext(
        transcript_bn=cleaned_transcript,
        spoken_place_bn=spoken_place_bn,
        spoken_place_en=spoken_place_en,
        spoken_district_en=spoken_district_en,
        spoken_plot_hint_bn=spoken_plot_hint_bn,
        spoken_crop=spoken_crop,
        spoken_crop_bn=spoken_crop_bn,
    )


def _transcribe_blocking(
    client: Any,
    audio_bytes: bytes,
    mime_type: str,
    settings: Settings,
) -> str | None:
    from google.genai import errors as genai_errors
    from google.genai import types

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    config = types.GenerateContentConfig(
        temperature=0.0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    last_exc: Exception | None = None
    for model_name in _candidate_models(settings):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[_TRANSCRIPTION_PROMPT, audio_part],
                config=config,
            )
            raw_text = (getattr(response, "text", None) or "").strip()
            return _clean_transcript(raw_text, settings.max_question_chars)
        except genai_errors.APIError as exc:
            last_exc = exc
            if exc.code in _RETRYABLE_GEMINI_CODES:
                log.warning(
                    "Audio transcription model %s returned HTTP %s; trying fallback.",
                    model_name,
                    exc.code,
                )
                continue
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            last_exc = exc
            log.warning(
                "Audio transcription model %s timed out (%s); trying fallback.",
                model_name,
                exc.__class__.__name__,
            )
            continue

    raise RuntimeError(f"All Gemini Flash models failed for audio: {last_exc}") from last_exc


async def transcribe_bangla_question(
    audio_bytes: bytes,
    mime_type: str,
    settings: Settings,
) -> str | None:
    """Transcribe a farmer's spoken Bangla question via Gemini Flash, or return None."""
    if not audio_bytes or not mime_type:
        return None

    api_key = settings.effective_gemini_api_key
    if not api_key:
        log.warning("Skipping voice-question transcription: Gemini API key is not configured.")
        return None

    try:
        client = _build_client(api_key, settings.audio_transcription_timeout_ms)
        return await asyncio.to_thread(
            _transcribe_blocking,
            client,
            audio_bytes,
            mime_type,
            settings,
        )
    except Exception as exc:  # noqa: BLE001 - voice input must never fail leaf analysis
        log.warning(
            "Voice-question transcription failed (%s); continuing without voice question.",
            exc.__class__.__name__,
        )
        return None


def _merge_with_offline(
    parsed: QuestionContext,
    question_text: str | None,
    *,
    has_audio: bool,
    max_chars: int,
) -> QuestionContext:
    transcript = (
        _clean_transcript(parsed.transcript_bn, max_chars)
        if (has_audio and parsed.transcript_bn)
        else None
    )
    combined_parts = [p for p in (question_text, transcript) if p and p.strip()]
    offline = extract_question_context_offline(
        " ".join(combined_parts) if combined_parts else None,
        transcript_bn=transcript,
    )

    place_bn = _clean_opt(parsed.spoken_place_bn) or offline.spoken_place_bn
    place_en = _clean_opt(parsed.spoken_place_en) or offline.spoken_place_en
    if place_bn and not place_en:
        place_en = _transliterate_bn_to_en(place_bn) or None

    district_en = _clean_opt(parsed.spoken_district_en) or offline.spoken_district_en
    plot_hint_bn = _clean_opt(parsed.spoken_plot_hint_bn) or offline.spoken_plot_hint_bn

    crop = parsed.spoken_crop
    crop_bn = _clean_opt(parsed.spoken_crop_bn, 60)
    if crop == "none" and offline.spoken_crop != "none":
        crop = offline.spoken_crop
        crop_bn = offline.spoken_crop_bn
    elif crop == "rice" and not crop_bn:
        crop_bn = "ধান"
    elif crop == "potato" and not crop_bn:
        crop_bn = "আলু"
    elif crop == "other" and not crop_bn:
        crop_bn = offline.spoken_crop_bn or "অন্য ফসল"
    elif crop == "none":
        crop_bn = None

    return QuestionContext(
        transcript_bn=transcript,
        spoken_place_bn=place_bn,
        spoken_place_en=place_en,
        spoken_district_en=district_en,
        spoken_plot_hint_bn=plot_hint_bn,
        spoken_crop=crop,
        spoken_crop_bn=crop_bn,
    )


def _extract_blocking(
    client: Any,
    audio_bytes: bytes | None,
    audio_mime: str | None,
    question_text: str | None,
    settings: Settings,
) -> QuestionContext:
    from google.genai import errors as genai_errors
    from google.genai import types

    has_audio = bool(audio_bytes and audio_mime)
    prompt = _EXTRACTION_PROMPT
    if question_text and question_text.strip():
        prompt = f"{_EXTRACTION_PROMPT}\n\nTyped farmer question: {question_text.strip()}"

    contents: list[Any]
    if audio_bytes and audio_mime:
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime)
        contents = [prompt, audio_part]
    else:
        contents = [prompt]

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QuestionContext,
        temperature=0.0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    last_exc: Exception | None = None
    for model_name in _candidate_models(settings):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            raw_text = (getattr(response, "text", None) or "").strip()
            if not raw_text:
                return extract_question_context_offline(question_text)
            try:
                data = json.loads(raw_text)
                parsed = QuestionContext.model_validate(data)
                return _merge_with_offline(
                    parsed,
                    question_text,
                    has_audio=has_audio,
                    max_chars=settings.max_question_chars,
                )
            except (json.JSONDecodeError, ValidationError):
                # Plain-text response (for example, a simplified transcript stub):
                # salvage transcript + offline extraction without wasting extra calls.
                plain_transcript = (
                    _clean_transcript(raw_text, settings.max_question_chars) if has_audio else None
                )
                combined = " ".join(
                    part.strip()
                    for part in (question_text, plain_transcript or raw_text)
                    if part and part.strip()
                )
                return extract_question_context_offline(combined, transcript_bn=plain_transcript)
        except genai_errors.APIError as exc:
            last_exc = exc
            if exc.code in _RETRYABLE_GEMINI_CODES:
                log.warning(
                    "Question context model %s returned HTTP %s; trying fallback.",
                    model_name,
                    exc.code,
                )
                continue
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            last_exc = exc
            log.warning(
                "Question context model %s timed out (%s); trying fallback.",
                model_name,
                exc.__class__.__name__,
            )
            continue

    raise RuntimeError(
        f"All Gemini Flash models failed for question context: {last_exc}"
    ) from last_exc


async def extract_question_context(
    audio_bytes: bytes | None,
    audio_mime: str | None,
    question_text: str | None,
    settings: Settings,
) -> QuestionContext:
    """Extract transcript, place, plot hint, and spoken crop via Gemini Flash or offline."""
    has_audio = bool(audio_bytes and audio_mime)
    has_text = bool(question_text and question_text.strip())
    if not has_audio and not has_text:
        return QuestionContext()

    api_key = settings.effective_gemini_api_key
    if not api_key:
        if has_audio:
            log.warning("Skipping voice-question extraction: Gemini API key is not configured.")
        return extract_question_context_offline(question_text)

    try:
        client = _build_client(api_key, settings.audio_transcription_timeout_ms)
        return await asyncio.to_thread(
            _extract_blocking,
            client,
            audio_bytes,
            audio_mime,
            question_text,
            settings,
        )
    except Exception as exc:  # noqa: BLE001 - voice/text parsing must never fail leaf analysis
        log.warning(
            "Question context extraction failed (%s); using offline fallback.",
            exc.__class__.__name__,
        )
        return extract_question_context_offline(question_text)
