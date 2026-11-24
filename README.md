# FosholLens (ফসল লেন্স)

**Multimodal Bangla Voice & Weather-Aware Crop Advisory for Rice and Potato**

Built for **CoU AI Hackathon 2026**.

---

## Team Tesseract

- **Prethu Raj Debnath**
- **Md. Kawsar Amin**
- **Nazim Saifullah** (`@nazimsaifullah`)

> **GitHub Contributor Transparency Note (Rule 3 / Rule 12):** Commits authored under `@niamt1` belong to a secondary GitHub account operated by registered Team Tesseract member **Nazim Saifullah** (`@nazimsaifullah`). All commits in this repository were authored solely by the three registered team members listed above.

---

## Problem & Solution Summary

### Problem
Smallholder rice (*ধান*) and potato (*আলু*) farmers in Bangladesh face severe yield losses from foliar diseases such as rice blast, bacterial leaf blight, brown spot, tungro, and potato early/late blight. Existing diagnostic tools typically return English-only labels, hallucinate unverified pesticide recommendations (even for bacterial or viral diseases where fungicides do not work), and ignore local weather—leading farmers to spray right before rainfall or strong winds.

### Solution
**FosholLens (ফসল লেন্স)** is a mobile-first Bangla web app and PWA that runs an end-to-end grounded crop advisory pipeline:
1. **ONNX Leaf Disease Vision** (CPU-only, local 8-class EfficientNet-B0) paired with a leaf-framing capture guide, explicitly flagging uncertain photos rather than fabricating diagnoses.
2. **Optional Bangla Voice & Text Question Input** allowing farmers to record a 12-second Bangla voice question or type a question alongside their photo, transcribed natively via Google Gemini Flash audio understanding.
3. **BARI/BRRI Grounded RAG Prescription Engine** (`16` curated chunks across all 8 leaf classes with precomputed Gemini embeddings + offline TF-IDF fallback), enforcing cultural-only guards for bacterial/viral diseases, no-spray healthy guidance, pre-harvest intervals (PHI), PPE reminders, and SAAO referrals.
4. **48-Hour Color-Coded Spray Timeline & GPS Auto-District** using Open-Meteo hourly forecasts across all 64 districts of Bangladesh (with one-tap Haversine GPS district matching), deterministically computing safe (`green`), caution (`amber`), and wash-off/drift risk (`red`) hours plus a pure-Bangla safest daylight spray window.
5. **Bangla Advisory Synthesis & Local Voice Playback** via Google Gemini Flash (with deterministic `TemplateWriter` fallback) and local ONNX Bangla TTS (`sulabhkatiyar/indic-tiny-tts-bengali` + `ffmpeg` -> MP3 by default for iOS Safari/Android/Desktop compatibility, with WAV fallback when `ffmpeg` is absent).

---

## Features & How the AI Works

The backend runs a 4-stage asynchronous job pipeline (`reading -> grounding -> writing -> speaking`), persisted in SQLite (WAL):

1. **Image Prep, Leaf Framing & Privacy (`apps/frontend` & `apps/api/app/routes.py`):**
   - `ImageDropzone.tsx` provides a dashed corner-framed leaf capture guide and Bangla lighting/framing tip.
   - Server-side preprocessing downscales photos to `1280px` max edge and re-encodes to JPEG, stripping EXIF metadata so phone GPS coordinates are never stored on disk.
   - Scope is **rice (`rice`) and potato (`potato`)** only—matching the crops the shipped EfficientNet checkpoint can name (`POST /api/v1/analyses` rejects other crops with `400 unsupported_crop`).
2. **Stage 1 — Vision & Voice Understanding (`apps/api/app/vision/`, `apps/api/app/llm/audio_input.py`):**
   - **Vision:** Runs local CPU ONNX inference (`onnxruntime`):
     - EfficientNet-B0 *(committed at `apps/api/models/foshollens_leaf.onnx`, 16 MB, ~40 ms)*: 8 rice & potato classes (`potato_early_blight`, `potato_healthy`, `potato_late_blight`, `rice_bacterial_leaf_blight`, `rice_blast`, `rice_brown_spot`, `rice_healthy`, and `rice_tungro`). Every label is surfaced to the farmer (`is_healthy` routes healthy plants to no-spray care).
     - Threshold (`VISION_CONFIDENCE_THRESHOLD=0.6`) is checked on the winning class's raw softmax probability, before crop filtering. A clear winner just below it is still accepted when it beats the runner-up by `VISION_MARGIN=0.30`, clears `VISION_MARGIN_FLOOR=0.45`, and the photo is at least `VISION_MARGIN_MIN_CROP_MASS=0.70` the selected crop.
   - **Optional Bangla Voice/Text Question:** Accepts optional `question_text` (up to `300` chars) and `audio` (up to `5 MB`; `audio/webm`, `audio/ogg`, `audio/mp4`, `audio/mpeg`, `audio/wav`, `audio/aac`, `audio/x-m4a`, `audio/x-wav`, `video/mp4`, `video/webm` with codec suffix stripping and MIME normalization). `transcribe_bangla_question()` sends audio bytes natively to Gemini Flash (`types.Part.from_bytes`) and degrades cleanly to `None` if unconfigured or silent.
3. **Stage 2 — BARI/BRRI Hybrid RAG & 48-Hour Weather Guard (`apps/api/app/grounding/`):**
   - **BARI/BRRI Grounded RAG (`rag.py`, `knowledge_base.py`):** Retrieves grounded Bangla/English passages from the 16-chunk corpus (`apps/api/data/bari_brri_corpus.json`) ranked by hybrid cosine similarity over precomputed `gemini-embedding-001` vectors (`apps/api/data/bari_brri_index.json`) and zero-network TF-IDF fallback—prioritizing the farmer's voice/text question when present.
     - Marks `rice_bacterial_leaf_blight` and `rice_tungro` as `cultural_only=True` & `refer_to_officer=True` (explicitly warning that fungicides do not work).
     - Marks `potato_healthy` & `rice_healthy` as no-spray healthy confirmations.
     - Marks uncertain diagnoses across any crop as non-chemical (`cultural_only=True`, `refer_to_officer=True`), retrieving safe scouting passages and advising against blind spraying.
   - **48h Weather Spray Timeline (`weather.py`, `districts.py`):** Evaluates Open-Meteo `forecast_days=3` sliced to the upcoming 48 hours (`rainProb`, precipitation `mm`, wind speed `km/h`; supporting exact GPS coordinates or district centroids and returning `None` if hourly data is empty or expired):
     - `red` (unsafe): `rainProb >= 60%`, `rainMm >= 1.0`, `windKmh >= 15.0`, or heavy rain (`>= 60%` or `>= 1.0 mm`) within the next 3 hours (wash-off lookahead).
     - `amber` (caution): `rainProb >= 30%`, `rainMm > 0.0`, or `windKmh >= 10.0`.
     - `green` (safe): clear conditions, computing a pure-Bangla `safestWindowSummaryBn` for the earliest daylight (`06:00–18:00`) safe window.
4. **Stage 3 — Grounded Bangla Advisory (`apps/api/app/llm/writer.py`):**
   - **Primary:** Google Gemini Flash (`gemini-3.5-flash-lite` default with fallback chain `gemini-3.8-flash` -> `gemini-3.7-flash` -> `gemini-3.5-flash` -> `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite` on HTTP 404/429/500/502/503/504 or timeout; optional `LLM_PROVIDER=openai` mode) synthesizes a 25–90 word colloquial Bangla advisory directly from retrieved BARI/BRRI passages, weather verdict, and farmer question.
   - **Safety Validation & Offline Fallback:** `validate_narrative()` rejects Latin letters, markdown, emojis, wrong word counts, or any chemical outside the retrieved BARI/BRRI chunks (blocking any chemicals when `cultural_only=True`). `TemplateWriter` composes grounded Bangla advice offline when no key is set or validation fails.
5. **Stage 4 — Local Bangla Voice Playback (`apps/api/app/tts/engine.py`):**
   - Synthesizes Bangla audio asynchronously via `sulabhkatiyar/indic-tiny-tts-bengali` (local ONNX, CPU) and transcodes via `ffmpeg` to compact MP3 audio (`TTS_FORMAT=mp3` default for universal iOS Safari/Android/Desktop playback; `opus` optional; WAV fallback if `ffmpeg` is unavailable). `prepare_text()` repairs `/` dose ratios and verifies medicine names survive before audio is served.

---

## Tech Stack, Models, APIs & Datasets

- **Frontend (`apps/frontend`):** React 19, TypeScript, Vite 7, React Router, Lucide icons, PWA manifest, Noto Sans Bengali, Browser `MediaRecorder` & Geolocation APIs
- **Shared Contracts (`packages/contracts`):** Pydantic v2 models (`apps/api/app/contracts.py`) mirrored in TypeScript/Zod (`@foshol-lens/contracts`)
- **Backend (`apps/api`):** Python 3.12+, FastAPI, Uvicorn, Pydantic v2, `aiosqlite` (SQLite WAL), HTTPX, Pillow, NumPy, `onnxruntime`, `huggingface-hub`
- **Vision Model:**
  - Shipped EfficientNet-B0 ONNX checkpoint (`apps/api/models/foshollens_leaf.onnx`, 8 rice & potato classes)
- **Embeddings & LLM / Audio AI:** Google Gemini (`google-genai` SDK) — `gemini-embedding-001` for precomputed RAG embeddings and `gemini-3.5-flash-lite` (with fallback chain) for Bangla voice transcription and advisory generation
- **Text-to-Speech:** [`sulabhkatiyar/indic-tiny-tts-bengali`](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-bengali) (local ONNX, CPU) + `ffmpeg` -> MP3 (default) / Opus / WAV fallback
- **Weather API:** [Open-Meteo Forecast API](https://open-meteo.com/) (key-free, non-commercial use)
- **Datasets & Agricultural Knowledge Sources:**
  - **Saon110 Bangladesh multi-crop plant disease dataset** (Hugging Face, CC BY-NC-SA 4.0) — the 5 rice classes
  - **PlantVillage** (Hughes & Salathé, 2015; Mohanty et al., 2016) — potato early blight, late blight, healthy (lab photos)
  - **Irish Potato Imagery Dataset** (Laizer & Mduma, 2025; Zenodo record 8286529, CC BY 4.0 per the paper) — field potato photos
  - **Paddy Doctor** (Petchiammal et al., 2022) — public rice dataset referenced by the merged Saon110 collection
  - **BARI** (*Bangladesh Agricultural Research Institute* — *BARI কৃষি প্রযুক্তি হাতবই*)
  - **BRRI** (*Bangladesh Rice Research Institute* — *BRRI আধুনিক ধানের চাষ*, *BRRI Knowledge Bank*) & **DAE** crop protection safety guidance

---

## Leaf Model: Training, Data & Evaluation

The shipped vision model is **EfficientNet-B0**, fine-tuned for 8 rice and potato leaf classes and exported to ONNX (`apps/api/models/foshollens_leaf.onnx`, 16 MB). It runs on CPU via `onnxruntime` at roughly 40 ms per photo. Training was done during the pre-build window on a free Kaggle GPU.

### Classes

| Crop | Class ID | Condition |
| --- | --- | --- |
| Rice | `rice_blast` | Blast (fungal) |
| Rice | `rice_brown_spot` | Brown spot (fungal) |
| Rice | `rice_bacterial_leaf_blight` | Bacterial leaf blight |
| Rice | `rice_tungro` | Tungro (viral, leafhopper-spread) |
| Rice | `rice_healthy` | Healthy |
| Potato | `potato_early_blight` | Early blight (fungal) |
| Potato | `potato_late_blight` | Late blight (water mould) |
| Potato | `potato_healthy` | Healthy |

### Data sources and licences

| Source | Used for | Licence | Photo type |
| --- | --- | --- | --- |
| Saon110 Bangladesh multi-crop plant disease dataset (Hugging Face) | The 5 rice classes | CC BY-NC-SA 4.0 | Merged public sources, mixed capture conditions |
| PlantVillage (Hughes & Salathé, 2015; Mohanty et al., 2016) | Potato early blight, late blight, healthy | CC BY-SA 3.0 as listed by the loader; not independently verified | Lab: single leaf on a plain background |
| Irish Potato Imagery Dataset (Laizer & Mduma, 2025; Zenodo record 8286529) | The same 3 potato classes | CC BY 4.0 per the paper; check the Zenodo record before any reuse | Field: phone photos taken on farms |

Two potato sources are used deliberately: a model trained only on lab photos mislabelled a field photo of a whole potato plant, and real field photos removed that failure. Paddy Doctor (Petchiammal et al., 2022) is credited as a public rice dataset referenced by the merged Saon110 collection.

**Licence note:** the Saon110 rice data is **non-commercial and share-alike**. This hackathon prototype complies; a commercial deployment would need different data or explicit permission.

### Splits

Every photo belongs to exactly one of three groups. The model learns from the training group, validation picks the best epoch, and the test group is scored once at the end.

| Class | Train | Validation | Test |
| --- | --- | --- | --- |
| Potato early blight | 2,000 | 250 | 250 |
| Potato healthy | 1,322 | 165 | 165 |
| Potato late blight | 2,000 | 250 | 250 |
| Rice bacterial leaf blight | 1,443 | 400 | 208 |
| Rice blast | 2,223 | 400 | 320 |
| Rice brown spot | 1,794 | 400 | 258 |
| Rice healthy | 1,234 | 352 | 178 |
| Rice tungro | 1,676 | 400 | 241 |
| **Total** | **13,692** | **2,617** | **1,870** |

PlantVillage has no leaf IDs, so its photos were split randomly. Near-identical photos can therefore land in both training and test groups, which we state as a leakage caveat rather than hide.

### Training recipe

- **Transfer learning:** start from ImageNet-pretrained EfficientNet-B0, replace the 1,000-way head with an 8-way head, and fine-tune the full network.
- **Input:** photos stored at up to 384 px on the long side, resized to **320 × 320** for the model.
- **Optimisation:** batch size 64, AdamW, learning rate `3e-4` with a one-cycle schedule, **25 epochs**.
- **Augmentation:** random resized crop (30–100% of the image), horizontal flip, rotation up to 25°, colour jitter, occasional blur, random erasing.
- **Regularisation:** dropout, stochastic depth, label smoothing, weight decay, and class weighting for the smallest class (`potato_healthy`).
- **Model selection:** validation accuracy is checked after every epoch and the best checkpoint is kept.
- **Export:** converted to ONNX and parity-checked against the PyTorch model (maximum difference below `2e-5`).
- **Hardware and time:** about 1 hour for 25 epochs on a free Kaggle GPU with mixed precision.

Architecture choice: EfficientNet-B0 (~5M parameters), EfficientNet-B3 (~12M) and ConvNeXt-Tiny (~28M) scored within noise of each other (98.4–98.6% on the 10-class runs), so the smallest and fastest model was shipped.

### Results (held-out test split)

**1,854 of 1,870 photos correct — 99.1% accuracy; macro F1 0.992.**

| Class | Precision | Recall | Test photos |
| --- | --- | --- | --- |
| Potato early blight | 1.000 | 0.992 | 250 |
| Potato healthy | 1.000 | 1.000 | 165 |
| Potato late blight | 0.992 | 1.000 | 250 |
| Rice bacterial leaf blight | 0.990 | 0.990 | 208 |
| Rice blast | 0.994 | 0.978 | 320 |
| Rice brown spot | 0.985 | 0.988 | 258 |
| Rice healthy | 0.994 | 0.989 | 178 |
| Rice tungro | 0.980 | 1.000 | 241 |

Rice blast is the weakest class (recall 0.978). Most of the 16 errors are blast, brown spot and tungro being confused with each other, which is expected because those lesions look alike.

### Known limitations (stated before judging)

- **Same-source test set.** Test photos come from the same public collections as training photos, so accuracy on new farm phone photos will be lower than 99.1%.
- **Lab versus field.** Part of the potato data is clean lab imagery. Field photos were added for potato, but not for every rice class.
- **Only 8 classes.** An unseen disease or another crop is forced into one of the 8. The crop selector, the `0.6` confidence threshold and the crop-mass warning reduce this risk but do not remove it.
- **Whole-photo classification.** The model labels the photo; it does not detect or outline lesions. A whole-plant photo can fool it, which is why the app asks for a close leaf photo.
- **Untuned app thresholds.** `VISION_CONFIDENCE_THRESHOLD=0.6`, `VISION_MARGIN=0.30`, `VISION_MARGIN_FLOOR=0.45` and `VISION_MARGIN_MIN_CROP_MASS=0.70` are sensible defaults, not values tuned on real farmer photos.
- **Decision support only.** The output is not a replacement for an SAAO or agriculture officer.

---

## AI Tools Disclosure

In accordance with **CoU AI Hackathon 2026 Rule 10**, Team Tesseract discloses the AI tools used in building FosholLens:
- **Antigravity:** Codebase exploration, feature implementation, refactoring, RAG/voice/timeline wiring, test execution, and build verification.
- **Claude (Anthropic):** System architecture design, safety guard review (Latin-script TTS checks, cultural-only chemical blocking, wash-off lookahead rules), documentation structure, and review of the leaf-model training plan (transfer learning, augmentation, data splits, ONNX export).
- **Google Gemini:** Prompt engineering for Bangla agricultural advisory & voice transcription, precomputing `gemini-embedding-001` corpus vectors, and runtime `gemini-3.5-flash-lite` inference (with Flash fallback chain).

No AI output is hard-coded, pre-recorded or faked (Rule 9.3). The leaf classifier is a trained EfficientNet-B0 ONNX network, the advisory text is generated at runtime from retrieved BARI/BRRI passages, and the deterministic weather verdict is computed in code rather than by a model.

---

## Third-Party Code & Assets Credits

### Software and libraries

- [`sulabhkatiyar/indic-tiny-tts-bengali`](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-bengali) — Local ONNX Bangla neural text-to-speech synthesis
- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), [NumPy](https://numpy.org/), and [ONNX Runtime](https://onnxruntime.ai/) — Backend framework, schema validation, vector math, and CPU inference
- [React](https://react.dev/), [Vite](https://vite.dev/), [React Router](https://reactrouter.com/), [Zod](https://zod.dev/), and [Lucide](https://lucide.dev/) — Frontend UI, routing, and contract validation
- [Noto Sans Bengali](https://fonts.google.com/noto/specimen/Noto+Sans+Bengali) — SIL Open Font License (OFL)
- [FFmpeg](https://ffmpeg.org/) — TTS audio transcoding to MP3/Opus (LGPL/GPL depending on build)
- [Open-Meteo API](https://open-meteo.com/) — Key-free hourly weather forecast API (non-commercial free tier)

### Model training stack

- [EfficientNet](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019) — Base architecture for the leaf classifier
- [PyTorch](https://pytorch.org/) and [timm](https://github.com/huggingface/pytorch-image-models) — Fine-tuning and model definition
- [ONNX Runtime](https://onnxruntime.ai/) — CPU inference in production
- [Kaggle](https://www.kaggle.com/) — Free GPU used for the 25-epoch training run

### Datasets

- **Saon110 Bangladesh multi-crop plant disease dataset** (Hugging Face) — **CC BY-NC-SA 4.0**. Rice classes are merged from several public sources, including the Paddy Disease Classification data. Non-commercial and share-alike.
- **PlantVillage** — Hughes & Salathé, 2015 ([arXiv 1511.08060](https://arxiv.org/abs/1511.08060)); Mohanty et al., 2016, *Frontiers in Plant Science* 7:1419, [doi 10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419). The loader lists CC BY-SA 3.0; we did not verify it independently. A Hugging Face mirror (`GVJahnavi/PlantVillage_dataset`) was used because the original loader failed.
- **Irish Potato Imagery Dataset** — Laizer & Mduma, 2025; [Zenodo record 8286529](https://zenodo.org/records/8286529); [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12020891/). The paper states CC BY 4.0; check the Zenodo record before reuse.
- **Paddy Doctor** — Petchiammal et al., 2022 ([arXiv 2205.11108](https://arxiv.org/pdf/2205.11108)). Public rice dataset referenced by the merged Saon110 collection.

### Agricultural knowledge

- Treatment guidance credited to **BARI**, **BRRI**, and **DAE**. The advisory is grounded in retrieved passages from the curated corpus, not invented by the model.

---

## Repository Structure

```text
foshollens/
├── apps/
│   ├── api/                           # Python 3.12+ / FastAPI service
│   │   ├── Dockerfile                 # Production container for Azure Container Apps (port 7860, HEALTHCHECK, UID 1000)
│   │   ├── .dockerignore
│   │   ├── pyproject.toml
│   │   ├── app/
│   │   │   ├── main.py                # FastAPI app, CORS, /health alias, lifespan startup checks
│   │   │   ├── routes.py              # /api/v1 health, districts, crops, analyses, image, audio
│   │   │   ├── contracts.py           # Pydantic API contracts (source of truth)
│   │   │   ├── config.py              # Environment settings (.env & apps/api/.env)
│   │   │   ├── store.py               # SQLite (WAL) persistence
│   │   │   ├── jobs/pipeline.py       # In-process 4-stage async job runner
│   │   │   ├── vision/                # ONNX leaf classifier (EfficientNet-B0)
│   │   │   ├── grounding/
│   │   │   │   ├── rag.py             # Hybrid vector + offline TF-IDF BARI/BRRI retriever
│   │   │   │   ├── knowledge_base.py  # Curated crop/disease KB & aliases
│   │   │   │   ├── weather.py         # Open-Meteo 12h verdict + 48h timeline + wash-off lookahead
│   │   │   │   └── districts.py       # 64 Bangladesh districts with coordinates
│   │   │   ├── llm/
│   │   │   │   ├── writer.py          # Gemini Flash + OpenAI-compat + validator + Template fallback
│   │   │   │   └── audio_input.py     # Gemini Flash native Bangla audio question transcription
│   │   │   └── tts/engine.py          # indic-tiny-tts-bengali ONNX + ffmpeg MP3/Opus encoder
│   │   ├── data/
│   │   │   ├── bari_brri_corpus.json  # 16-chunk curated 8-class BARI/BRRI/DAE knowledge corpus
│   │   │   └── bari_brri_index.json   # Precomputed 768-dim Gemini embeddings
│   │   ├── models/
│   │   │   └── foshollens_leaf.onnx   # Committed 8-class EfficientNet-B0 ONNX checkpoint (16 MB)
│   │   ├── scripts/
│   │   │   └── build_rag_index.py     # Rebuilds bari_brri_index.json via gemini-embedding-001
│   │   └── tests/                     # pytest tests (API, RAG, voice, weather, vision, TTS, CORS)
│   └── frontend/                      # @foshol-lens/frontend (React 19 + TypeScript + Vite PWA)
│       ├── vercel.json                # Standalone SPA rewrite config
│       ├── index.html
│       ├── public/
│       │   ├── favicon.svg
│       │   └── manifest.webmanifest
│       └── src/
│           ├── components/            # Upload, Dropzone framing guide, WeatherCard timeline, Result, Audio
│           ├── pages/                 # HomePage, HistoryPage, AnalysisPage
│           ├── lib/                   # API client, GPS district matcher, image prep, Bangla formatting
│           └── styles/global.css
├── packages/
│   └── contracts/                     # @foshol-lens/contracts (Zod schemas & shared routes)
├── docs/
│   └── backend-architecture.md
├── scripts/free-port.sh
├── .env.example
├── .gitignore
├── package.json
├── vercel.json                        # Root Vercel monorepo build & SPA rewrite config
└── README.md
```

---

## Local Setup

### Prerequisites
- **Node.js** `>= 22` and `npm`
- **Python 3.12** (recommended via [`uv`](https://docs.astral.sh/uv/))
- **`ffmpeg`** on `PATH` (optional, required for Bangla MP3/Opus TTS audio output; falls back to WAV when absent)

### 1. Environment Variables
Copy `.env.example` to `.env` in the repo root (or `apps/api/.env`). Never commit `.env`:

```bash
cp .env.example .env
```

Set `GEMINI_API_KEY` to enable Google Gemini Flash Bangla advisory generation and voice-question transcription. Without an API key, the app still runs end-to-end using offline TF-IDF RAG retrieval and `TemplateWriter` fallback (`generatedBy: "template"`).

### 2. Install Dependencies

```bash
# Root JS workspaces (@foshol-lens/contracts + @foshol-lens/frontend)
npm install

# Python API environment (apps/api)
cd apps/api
uv venv --python 3.12 .venv
# Linux/macOS:
uv pip install --python .venv/bin/python -e . pytest pytest-asyncio ruff mypy
# Windows PowerShell:
# uv pip install --python .venv/Scripts/python.exe -e . pytest pytest-asyncio ruff mypy
cd ../..
```

*(Optional)* Rebuild the precomputed RAG embedding index after editing `apps/api/data/bari_brri_corpus.json`:

```bash
cd apps/api && .venv/bin/python scripts/build_rag_index.py
```

### 3. Run Locally

Start the API (`:4000`) and frontend (`:5173`) in separate terminals:

```bash
# Terminal 1 — FastAPI backend (port 4000)
cd apps/api
.venv/bin/uvicorn app.main:app --reload --port 4000
# Windows PowerShell: .venv\Scripts\uvicorn.exe app.main:app --reload --port 4000

# Terminal 2 — Frontend + Vite dev server (port 5173)
npm run dev:frontend
```

Open `http://localhost:5173` and verify backend health at `http://localhost:4000/api/v1/health`.

### 4. Tests & Builds

```bash
# Backend pytest, ruff, and mypy
cd apps/api && .venv/bin/python -m pytest -v
cd apps/api && .venv/bin/ruff check app tests scripts

# Contracts & Frontend production build
npm run build
npm run typecheck
```

---

## Deployment

### Backend on Azure Container Apps (Docker)
1. Build and push [`apps/api/Dockerfile`](apps/api/Dockerfile) using build context `apps/api` (installs `ffmpeg`, copies `models/foshollens_leaf.onnx` & `data/`, runs as UID `1000`, exposes port `7860` with a built-in Docker `HEALTHCHECK` on `/api/v1/health`, and starts Uvicorn on `0.0.0.0:${PORT:-7860}`; `/health` is also exposed as a root alias).
2. Configure the Container App with:
   - **External Ingress Target Port:** `7860`
   - **Scaling:** `minReplicas=1` (or `0`) and **`maxReplicas=1`** — required so all requests share a single SQLite (WAL) database and local `/app/storage` volume safely.
   - **Resources:** `1.0 vCPU` / `2.0 Gi` RAM (or `2.0 vCPU` / `4.0 Gi` RAM) for CPU ONNX vision + local TTS synthesis.
3. Configure Container App Environment Variables / Secrets:
   - `APP_ENV=production`
   - `GEMINI_API_KEY=<your-gemini-api-key>`
   - `CORS_ORIGINS=http://localhost:5173,https://<your-vercel-domain>.vercel.app`
   - `PUBLIC_BASE_URL=https://<your-container-app-fqdn>`

### Frontend on Vercel
1. Import the repository into Vercel with the **Root Directory** left at the repository root (`.`). Root [`vercel.json`](vercel.json) runs `npm ci`, builds `@foshol-lens/contracts` and `@foshol-lens/frontend`, serves `apps/frontend/dist`, and rewrites SPA routes (`/analysis/:id`, `/history`).
2. Set the Vercel environment variable:
   - `VITE_API_BASE_URL=https://<your-container-app-fqdn>`

---

## Links

| Item | Value |
| --- | --- |
| GitHub repository | This repository (public) |
| Live deployment link | _[To be added after deployment]_ |
| Demo video (max 3 minutes) | _[To be added before submission]_ |
| Test login credentials | Not applicable — the app has no login or user accounts |

---

## Rulebook Compliance Checklist (CoU AI Hackathon 2026)

Required README content, mapped to where it lives in this document:

| Requirement | Rule | Where |
| --- | --- | --- |
| Project, team and member names | 10.2 | [Team Tesseract](#team-tesseract) |
| Problem and solution summary | 10.2 | [Problem & Solution Summary](#problem--solution-summary) |
| Features and how the AI works | 10.2 | [Features & How the AI Works](#features--how-the-ai-works) |
| Tech stack, models, APIs, datasets | 9.2, 10.2 | [Tech Stack, Models, APIs & Datasets](#tech-stack-models-apis--datasets), [Leaf Model](#leaf-model-training-data--evaluation) |
| AI tools used, and for what | 9.1, 10.2 | [AI Tools Disclosure](#ai-tools-disclosure) |
| Third-party code and assets, credited | 8.4, 10.2, 14 | [Third-Party Code & Assets Credits](#third-party-code--assets-credits) |
| Open-source libraries and public APIs listed | 8.4 | [Tech Stack](#tech-stack-models-apis--datasets), [Credits](#third-party-code--assets-credits) |
| Public or licensed data credited | 9.5 | [Data sources and licences](#data-sources-and-licences), [Datasets](#datasets) |
| Real AI, no faked or pre-recorded output | 9.3 | [AI Tools Disclosure](#ai-tools-disclosure) |
| Setup steps | 10.2 | [Local Setup](#local-setup) |
| Live link and demo video | 10.2, 11, 12 | [Links](#links) |
| Test login credentials (if the app has a login) | 12.3 | [Links](#links) — not applicable |
| `.env.example`, no committed keys | 10.2 | [`.env.example`](.env.example), [Local Setup](#local-setup) |

**Outstanding before the 24 Nov deadline:** replace the live deployment link and demo video placeholders in [Links](#links) with public URLs. A broken or private link counts as missing (Rule 12).
