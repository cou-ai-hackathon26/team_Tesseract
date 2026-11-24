# ফসল লেন্স — Backend Architecture

Status: **built.** Phases 0–3 are implemented in `apps/api` and run end to end.
Vision (§4) ships as an ONNX classifier with deliberately narrow coverage; when
it cannot place a leaf the result says so rather than guessing.

---

## 1. The pipeline

Four stages, each with one owner and one failure mode:

```
  image ──► [1] READING (deferred: no classifier yet)
              │  structured report: crop, disease, confidence, top-k
              ▼
            [2] GROUNDING (our code, deterministic)
              │  curated BARI/BRRI treatment facts + 48h weather for the district
              ▼
            [3] LLM (OpenAI-compatible API, your token)
              │  short friendly Bangla narrative — phrasing only, no new facts
              ▼
            [4] TTS (our hardware)
                 Bangla audio for the narrative
```

The important property: **stage 2 is the only thing allowed to state a fact.**
Stage 3 rewrites prose. It never introduces a pesticide, a dose, or a symptom
that wasn't handed to it. That's what keeps a hallucination from telling a
farmer to spray the wrong chemical.

**Stage 1 reads the photo, but refuses more often than it confirms.**
`vision/classifier.py` runs an 8-class EfficientNet-B0 and maps every label to a
curated knowledge-base entry. A below-threshold probability, or a label belonging
to a different crop, produces `undetermined`; the pipeline then falls back to a
per-crop KB entry and marks the result unconfirmed. A confident in-crop label —
including `*_healthy` — is surfaced as `identified`. That fallback is the same
`FALLBACK_DISEASE` lookup as before, now one branch instead of the only branch.

---

## 2. Process boundaries

**One runtime, one process.** Python / FastAPI.

```
apps/api
  ├─ HTTP layer ......... upload, job status, result, audio, media
  ├─ job runner ......... in-process asyncio task + semaphore
  ├─ TTS ................ ONNX Runtime, in-process
  ├─ grounding .......... KB lookup + weather fetch
  └─ LLM ................ openai SDK → OpenAI-compatible endpoint
```

There is no localhost HTTP hop between the API and the TTS model.

The old `apps/backend` (Node/Express, health endpoint only) has been deleted.
Nothing depended on it.

The frontend stays TypeScript.

### Verified hardware

Checked on this machine, not assumed:

|        |                                                         |
| ------ | ------------------------------------------------------- |
| CPU    | AMD Ryzen 7 5700U, 8 cores / 16 threads, up to 4.37 GHz |
| RAM    | 14 GiB total, **~3 GiB available**                      |
| Swap   | 15 GiB, **6.9 GiB already in use**                      |
| GPU    | Radeon integrated (Lucienne) — **no CUDA, no ROCm**     |
| Disk   | 44 GiB free                                             |
| ffmpeg | 9.0.2                                                   |
| Python | 3.14.10 system; 3.12/3.13 available via `uv`            |

**Consequence: CPU-only, and memory-constrained.** Two decisions follow:

1. **Everything runs quantised on CPU.** Vision via ONNX Runtime int8; TTS via
   ONNX Runtime. No PyTorch, no CUDA dependency. This is not a compromise —
   §7's model is measurably 9–12× real time here.
2. **Use Python 3.12, pinned via `uv`**, not the system 3.14. `onnxruntime`
   resolves on 3.14, but ML wheels lag new interpreters and a surprise
   incompatibility mid-hackathon is not worth it. `uv python pin 3.12` makes it
   reproducible without touching the system Python.

Budget at idle: TTS ~300 MB peak (measured). Vision similarly small if
EfficientNet-B0/MobileNet are used. Total process well under 1 GB — which
matters, because this box has ~3 GB free and is already swapping.

### Concurrency — decide this up front

Python is the right call here, but this is the one place it can hurt, and it is
much cheaper to design for than to discover under load.

- **Never `uvicorn --workers N`.** Each worker loads its own copy of the model
  weights. Two workers roughly doubles resident memory for zero throughput gain.
  One worker, models loaded once at startup.
- **Never call a model on the event loop.** Inference is blocking and CPU-bound;
  it stalls every other request, including `/health`.
- **Run inference in a thread pool** (`anyio.to_thread` / `run_in_threadpool`).
  Python threads are normally GIL-bound, but ONNX Runtime releases the
  GIL during native compute, so they genuinely parallelize here.
- **Cap concurrent inference with a semaphore at ~2**, not `os.cpu_count()`.
  Beyond that, contention makes every request slower without raising throughput.
- **TTS concurrency 1.** It is the heaviest stage; serialising it keeps latency
  predictable, and the job model means it is a real queue anyway.
- **SQLite access via `aiosqlite`** or off-loop, for the same reason as models.

None of this leaks into the HTTP layer. Jobs are a module, so if a single box is
ever outgrown the runner moves to `arq` (Redis) or a separate worker process
without touching routes or the frontend.

---

## 3. Job model — the analysis is not one request

Measured budget: vision 150–400 ms, LLM 2–5 s, TTS 1–4 s. A blocking request
would sit open for 10+ seconds and die on a flaky mobile connection. The
contract already has `pending | processing | completed | failed` and a
summary/result split — it was shaped for exactly this.

```
POST /analyses ──► 202 { id, status: 'pending' }
                         │
GET /analyses/:id ──► { status, stage }        ← poll every 700ms
                         │
                     completed ──► GET /analyses/:id/result → text
                                   GET /analyses/:id/audio  → 202 | audio
```

**Stages** (as implemented in `contracts.py`):

```
queued → reading → grounding → writing → speaking → done
```

Surfacing `stage` lets the loading UI say "ফসলের তথ্য খোঁজা হচ্ছে…" then
"আপনার জন্য পরামর্শ লেখা হচ্ছে…" instead of an anonymous spinner. It's cheap and
it makes the wait feel half as long.

**Text lands before audio, and the stage keeps moving after it.** `status:
completed` fires as soon as the narrative is stored, while `stage` is still
`speaking`. Audio is a separate subresource with its own state, synthesised in a
detached task; when it finishes, the row advances to `done` and the stored
`result_json.stage` is rewritten too. That last part matters: `GET .../result`
serves `result_json`, so a stored result still saying `speaking` would leave the
client reading a finished analysis that looked unfinished.

The audio player therefore polls `/result` while its clip is `pending` and swaps
itself in when the file appears. `preload="none"` means it costs no data until
tapped.

---

## 4. Stage 1 — Vision (built, narrow)

`vision/classifier.py` runs one ONNX checkpoint, served through `onnxruntime` —
already a dependency, so it adds no install weight. Everything that differs
between checkpoints lives in `vision/models.py` as a spec: label order, crop and
knowledge-base mapping, input size, normalisation and input node name.

| Arch            | Classes | Crops        | Size  | Latency  | Source          |
| --------------- | ------- | ------------ | ----- | -------- | --------------- |
| EfficientNet-B0 | 8       | rice, potato | 16 MB | 28–49 ms | local `models/` |

Measured on this box, CPU only. The checkpoint is committed
(`apps/api/models/foshollens_leaf.onnx`, 16 MB), so a fresh clone runs with
vision and needs no extra setup. It is the one file exempted from the
`apps/api/models/` ignore rule; scratch exports and other checkpoints stay out of
the repo.

Two deliberate consequences:

* `/health`'s `vision` field asks the classifier, not the env flag. A local
  checkpoint that is absent reports `unavailable`; before this it reported `ready`
  and every upload then degraded to "could not identify" with nothing to explain
  why.
* Startup logs a warning naming the missing path when the checkpoint is absent,
  so the problem is visible in the log rather than discovered one failed upload
  at a time. A missing file still degrades the stage rather than failing the job:
  the curated advice behind it is still true.

**Why ONNX and not `transformers`:** the upstream PyTorch path costs a ~900 MB
`torch` wheel for a 16 MB model. The ONNX export produces the same logits from
the same weights with no torch at all.

**Preprocessing is part of the model contract.** Input size (320), normalisation
(ImageNet mean/std) and the ONNX input node name (`input`) are all in the spec.
Getting any of them wrong does not raise — it silently degrades accuracy — which
is why they live next to the label list rather than as module constants.

**Threshold placement is the whole safety story.** `_decide` follows
`predict_final.py`: it masks to the requested crop and renormalises for ranking
only, then compares the winning class's **raw, unmasked** softmax probability
against the threshold. Thresholding the renormalised value would let a
wrong-crop photo clear the bar, because renormalisation discards the evidence
that the photo was mostly another crop.

A second acceptance path handles the "clear photo, just under one number" case:
a winner below `VISION_CONFIDENCE_THRESHOLD` is still accepted when it beats the
runner-up by `VISION_MARGIN`, clears `VISION_MARGIN_FLOOR`, and the photo is at
least `VISION_MARGIN_MIN_CROP_MASS` the selected crop. On the local test sets a
flat threshold cut mostly admitted confident mistakes, while this strict margin
rule rescued a correct answer without adding wrong ones.

`undetermined` is returned for any of:

| Condition                      | Why                                                  |
| ------------------------------ | ---------------------------------------------------- |
| raw probability < 0.6          | not enough evidence                                  |
| label's crop ≠ requested crop  | wrong photo, or a crop this model cannot name        |
| label has no KB entry          | no curated dose exists; inventing one is the failure |
| model unavailable / unreadable | no evidence at all                                   |

Every class the checkpoint emits has a curated KB entry: four rice diseases,
two potato diseases, and the two `*_healthy` classes. A confident `*_healthy`
label is a real answer, not a refusal: it routes to a no-spray care entry. The
checkpoint spells its bacterial-blight class `rice_bacterial_leaf_blight` while
the KB id is `rice_bacterial_blight`; the mapping is explicit rather than string
surgery, and a test asserts every mapped id exists in the knowledge base.

**What the unconfirmed path does and does not do.** It keeps the crop's organic
care steps and its source, and it withholds every chemical dose, because a dose
is specific to a disease nobody confirmed. The LLM receives no chemical list in
that case, so it cannot phrase one either. The result carries
`uncertainty.isUncertain = true` and `diagnosis.confidence = null` — the label
shown is the fallback entry and no number is attached to it.

**The remaining problem is still domain shift.** PlantVillage is studio photos on
a uniform background; farmers shoot a whole plant in shade. On the stock photos
tested here the model missed most field-style photos. Mitigations, in order of
value:

1. Collect 50–100 real field photos from the target crops and measure before
   trusting anything. Budget for this; it's the actual work.
2. **Temperature-scale the confidence** on a held-out set. Raw softmax is
   overconfident; an uncalibrated 0.87 is meaningless.
3. The refusal threshold above is in place; tune `VISION_CONFIDENCE_THRESHOLD`
   once field data exists.
4. Reject-then-ask: if the top-2 are within 0.05, also refuse.

**The real problem is not the model, it's the domain shift.** PlantVillage is
studio photos on a uniform background. Farmers shoot a whole plant, in shade,
with a phone, at an angle. Accuracy collapses. Mitigations, in order of value:

1. Collect 50–100 real field photos from the target crops and check the model
   against them before trusting anything. Budget for this; it's the actual work.
2. **Temperature-scale the confidence** on a held-out set. Raw softmax is
   overconfident; an uncalibrated 0.87 is meaningless. After scaling, the number
   means what the UI claims it means.
3. A refusal threshold. Below `τ_low` (start ~0.45, tune) return `uncertain` and
   route to "কৃষি কর্মকর্তার পরামর্শ নিন" instead of guessing.
4. Reject-then-ask: if the top-2 are within 0.05, treat as ambiguous.

**Crop coverage is a scoping decision, not a bug.** PlantVillage gives rice,
potato, tomato, chili, maize well, but the shipped EfficientNet checkpoint is
trained on rice and potato only. The app therefore offers rice and potato and
rejects anything else with `unsupported_crop`: a crop no model can name is a
dead end for the farmer, not a feature. Extend the enum in the contract types
only when a checkpoint actually supports it.

**Do not send the image to the LLM in v1.** Perception belongs to the vision
model; language belongs to the LLM. Giving both the image invites the LLM to
"correct" the classifier from a compressed thumbnail, and there's no way to
audit which one won. Revisit as a _low-confidence second opinion_ only — one
extra call, gated on `confidence < τ`, logged separately.

**Grad-CAM (optional, high demo value):** one heatmap from the last conv layer,
served as a PNG overlay. It gives the "detailed report" something visual and it
is genuinely useful for spotting a model keying on background instead of leaf.

---

## 5. Stage 2 — Grounding (deterministic, no model)

Two inputs, both looked up in code:

### 5a. Treatment facts

A curated JSON keyed by `crop × disease`, sourced from BARI/BRRI
recommendations — the file `TREATMENTS_DATA_PATH` already anticipates.

```json
{
  "rice": {
    "blast": {
      "pathogen": "Magnaporthe oryzae",
      "symptoms": ["..."],
      "organic": ["..."],
      "chemical": [{ "product": "...", "dose": "...", "interval": "..." }],
      "prevention": ["..."],
      "source": "BRRI Adhunik Dhaner Chash, 2023"
    }
  }
}
```

This store, not the LLM, feeds the `treatment` and `prevention` arrays in the
result contract. It also carries a `source` string we can show, which is worth
more to a farmer than any amount of model prose.

### 5b. Weather — Open-Meteo (free, verified)

You said the LLM searches weather. **I'd push back, and here's the concrete
reason:** a fabricated "আগামীকাল বৃষ্টি হবে, আজই স্প্রে করুন" is worse than no
weather sentence at all, because weather is exactly what decides whether a spray
is wasted. The stall is the same one the whole design avoids.

So: fetch it ourselves, hand it over as facts, let the LLM only phrase it.

**Verified against the live API** (`api.open-meteo.com/v1/forecast`):

|                |                                                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| Cost           | Free, **no API key**, no signup                                                                                      |
| Limits         | 600 calls/min, 5,000/hour, 10,000/day (confirmed by response headers)                                                |
| Commercial use | **Not licensed** on the free tier                                                                                    |
| Attribution    | Required, CC-BY-4.0                                                                                                  |
| Data           | 48h hourly: `precipitation_probability`, `precipitation`, `wind_speed_10m`, `relative_humidity_2m`, `temperature_2m` |
| Latency        | 69 ms, `timezone=Asia/Dhaka` handled server-side                                                                     |

Measured for Rajshahi, 48 hours returned starting `2026-11-20T00:00`:

```
00:00  rain%=18  mm=0.0  wind=5.7   rh=90%
03:00  rain%=76  mm=1.8  wind=6.9   rh=96%
06:00  rain%=69  mm=0.5  wind=10.2  rh=97%
12:00  rain%=90  mm=0.5  wind=8.8   rh=88%
```

That is exactly the shape the spray-window rule needs.

**On "commercial use: not licensed":** the free tier is for evaluation and
prototyping, which is what this is. If this ever ships to real farmers, either buy
the Standard plan (1M calls/month, commercial licence) or self-host — the server
is open source. Budget it as a real line item rather than assuming free forever.
At 64 districts × 48 refreshes/day that is ~3,000 calls/day, comfortably inside
the limit but nowhere near a commercial licence.

- **Location:** a district picker (64 districts, saved locally), not
  geolocation. One tap, no permission dialog, works indoors, and coarse enough to
  be uninteresting to anyone surveilling. Geolocation as opt-in later, if ever.
- Cache per district for 30 minutes — weather doesn't change that fast and it
  keeps us off the API.
- Timeout 2.5 s; on failure the weather block is absent and the LLM is told not to
  mention weather at all.

**Spray-window rule** computed in code, never by the LLM: rain probability > 60%
in the next 12h, or wind > 15 km/h → `sprayWindowOk: false` with the reason.
Deterministic verdict, LLM-provided wording. If you _do_ want tool-calling
weather, keep this rule as the gatekeeper — the LLM may explain the verdict,
never produce it.

---

## 6. Stage 3 — LLM

**Provider: Tokens BD**, OpenAI-compatible, verified live:

```
base_url = https://tokens.bd/v1
model    = provider/model-id      (e.g. "deepseek/deepseek-v4.1-flash")
auth     = Authorization: Bearer tok_live_...
```

`GET /v1/models` returns `401 missing_api_key` without a key, so the gateway is
up and auth is enforced. Use the official `openai` Python SDK — it is the same
call shape the provider documents, and we already need `httpx` for the weather
fetch.

### Provider constraints that shape the code

These come from the provider's own docs and are load-bearing:

| Limit               | Value                                                       | Consequence                                |
| ------------------- | ----------------------------------------------------------- | ------------------------------------------ |
| Requests per minute | 60 (default)                                                | Fine — one analysis is one call            |
| Concurrency         | 10 with a plan, **3 without**                               | Cap our in-flight LLM calls at ≤3          |
| Body size           | 10 MB                                                       | Images never go to the LLM anyway          |
| Rate-limit errors   | 429 `rate_limited`, `concurrency_limit`, `window_exhausted` | `window_exhausted` must **not** be retried |
| Spend caps          | 403 `monthly_spend_cap_exceeded`                            | Not retryable; fall back to template       |

So: `max_retries=0` in the SDK and explicit handling, because the SDK's default
retry would burn a quota that resets in hours.

### Structured output — use the forced-tool path

The gateway does **not** enforce or repair JSON. `response_format` is forwarded
only when the upstream provider speaks OpenAI format; if the model is served over
Anthropic format, the gateway translates the request and **silently drops**
`response_format`, `n` and `seed`. Nothing tells you.

The provider's docs are explicit that **a forced tool call is the portable
method** — it survives translation in both directions. So:

```python
tools = [{
  "type": "function",
  "function": {
    "name": "speak_advice",
    "strict": True,
    "parameters": {
      "type": "object",
      "properties": {"spoken": {"type": "string"}},
      "required": ["spoken"],
      "additionalProperties": False,
    },
  },
}]
resp = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[...],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "speak_advice"}},
    temperature=0.4,
    max_tokens=400,
    max_retries=0,
)
spoken = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)["spoken"]
```

`max_tokens=400`, not 220: **Bengali takes more tokens than English for the same
content** (the provider documents this). A limit tuned in English truncates a
Bangla answer. Always check `finish_reason` — `length` means the JSON is
mutilated and will not parse.

### Input — structured JSON, never prose

```json
{
  "diagnosis": { "crop": "ধান", "disease": "ব্লাস্ট", "confidence": "উচ্চ" },
  "symptoms_observed": ["..."],
  "treatment": { "organic": ["..."], "chemical": [{ "product": "...", "dose": "..." }] },
  "prevention": ["..."],
  "weather": { "rain_next_12h_prob": 70, "spray_window_ok": false, "reason": "বৃষ্টির সম্ভাবনা" },
  "uncertain": false
}
```

Confidence goes in as a **band** (`উচ্চ` / `মাঝারি` / `নিম্ন`), not `0.87`. A
number invites the model to repeat a precise-sounding figure the farmer will
over-trust, and it reads badly aloud.

### Output — one field, and it must be Bangla script

```json
{ "spoken": "..." }
```

The cards, steps and doses come from the knowledge base, so the LLM cannot
corrupt them, and the narrative cannot drift from the structured result the UI
shows. One string to validate, one string to speak.

**System prompt rules (drafted in Bangla, enforced by a validator after):**

- Answer in **Bengali, Bangla script, always**. Say it in English _and_ Bangla —
  the provider notes models otherwise default to English or romanise.
- 3–4 sentences, 40–70 words. It is being read aloud; long is unusable.
- Spoken register — as one farmer talks to another (`আপনি`, not `মহোদয়`).
- No markdown, no bullets, no emoji, no URLs, **no Latin script**.
- Only the medicines and doses in `treatment.chemical`. Nothing else, ever.
- If `uncertain`, say so plainly and point to the upazila agriculture officer.
- If `weather` is null, don't mention weather.
- Bangla numerals (০-৯) only.

### Post-validation, deterministic — and non-negotiable

Reject and retry **once**, then fall back. Reject if:

| Check                                      | Why                                 |
| ------------------------------------------ | ----------------------------------- |
| Fails to parse / `finish_reason == length` | truncated JSON                      |
| Contains Latin letters `[A-Za-z]`          | TTS **silently drops** it (§7)      |
| Contains markdown or emoji                 | TTS reads punctuation aloud         |
| Word count out of bounds                   | unusable when spoken                |
| Names a chemical absent from the facts     | safety                              |
| No Bangla characters at all                | wrong language despite instructions |

That Latin check is not cosmetic. If the model writes `Tricyclazole`, the TTS
normaliser deletes the word and produces a fluent sentence that omits the actual
medicine — the single most dangerous failure in this whole pipeline.

Normalise to **NFC** before comparing or caching (the provider flags `য়` and
`কো` having two encodings).

### Fallback

If the API is down, rate-limited, over a spend cap, or returns unusable output
twice: build the narrative from a template over the same facts —
`generatedBy: "template"`. The analysis still succeeds; it just reads more
plainly. A demo that dies because a quota rolled over is avoidable.

---

## 7. Stage 4 — TTS

**Chosen model: `sulabhkatiyar/indic-tiny-tts-bengali`.** I downloaded it and ran
it on this machine rather than trusting the model card.

| Measured on this box (Ryzen 7 5700U, 16 threads, CPU only) | Value                       |
| ---------------------------------------------------------- | --------------------------- |
| Model size                                                 | 22.5 M params, ~90 MB ONNX  |
| Load time                                                  | 0.27 s                      |
| Synthesis, 7.8 s of audio                                  | **0.83 s** (RTF ≈ 0.11)     |
| Synthesis, male voice, 7.2 s                               | 0.61 s (RTF ≈ 0.084)        |
| Peak RSS (loaded + 5 synth)                                | **298 MB**                  |
| Voices                                                     | `female`, `male`            |
| Output                                                     | 24 kHz mono WAV             |
| Licence                                                    | CC-BY-4.0 weights, MIT code |

So it is ~9–12× faster than real time with no GPU and no PyTorch — the ONNX path
needs only `onnxruntime` + `numpy`. On this hardware that is the deciding fact:
XTTS-v2 (~1.8 GB, CUDA) is not merely worse, it is **not available**, since there
is no CUDA and no ROCm. MMS-TTS was the previous pick; this model scores 2.1 %
CER vs MMS-TTS's 4.86 % on FLORES-200 and is smaller, so it wins outright.

### Two safety bugs found in its normaliser — must be handled

The model ships `normalize_for_inference(text, "bn")`. It is good, but two of its
behaviours are dangerous for a pesticide app, and I verified both by running it:

```
"২ মিলি/লিটার"  →  "দুই মিলি বা লিটার"        ← "or" !
```

A slash means _per_, not _or_. `২ মিলি/লিটার` is a dose ratio; spoken as
"দুই মিলি **বা** লিটার" it becomes nonsense that could be heard as "2 ml or a
litre". **Always expand `/` to "প্রতি" in our own pre-pass**, before handing text
to the model.

```
"Tricyclazole ব্যবহার করুন"  →  "ব্যবহার করুন"   ← chemical silently dropped
```

Any Latin-script text is **removed**. If the LLM ever writes a pesticide name in
Latin, the audio would quietly omit the most important word in the sentence — and
worse, it would sound like a complete instruction while being incomplete. This is
why §6's validator must reject Latin script _before_ TTS, and why the KB must
store a Bangla transliteration for every product (`ট্রাইসাইক্লাজল`), not just the
trade name.

So the pipeline gets its own `prepare_text()` **before** the model's
normaliser: expand `/` → প্রতি, spell out ranges and hyphens, strip markdown,
reject any surviving Latin script, and **assert that a named chemical term is
still present**. If the assertion fails, synthesis raises `TtsSafetyError` and
the audio is marked `failed` rather than shipping a mutilated instruction.

One subtlety the first implementation got wrong: the assertion must only require
terms the narrative actually _contains_. Demanding every candidate chemical in
the KB entry refused valid audio, because a narrative recommends one product, not
all of them.

**Audio format is the real constraint, not the voice.** Measured on a real 7.8 s
narrative:

| Format           | Size      |
| ---------------- | --------- |
| WAV 24 kHz       | 367 KB    |
| MP3 32 kbps      | 31 KB     |
| **Opus 24 kbps** | **22 KB** |

16× smaller. Transcode with the installed `ffmpeg` 9.0.2 to Opus, MP3 fallback
for older Android. A farmer on 2G waiting for 367 KB will not wait.

**Cache by hash.** Key = `sha256(text + voiceId + format)`. Bangla narratives
repeat heavily, so hit rate is high and synthesis is skipped on repeat.

**Bangla script only.** The model verbalises digits correctly — `১০ লিটার` →
"দশ লিটার", `80%` → "আশি শতাংশ", `₹৫০০` → "পাঁচশো টাকা", `১০:৩০` → "দশটা
ত্রিশ মিনিট" — but it cannot read Latin at all, so §6's system prompt must
demand Bangla script and the validator must enforce it.

---

## 8. Contract changes

Additive, so the mock client keeps compiling. The source of truth moves from
hand-written zod to pydantic models in `apps/api`, with TypeScript generated from
the spec FastAPI emits:

```
apps/api/app/contracts.py   pydantic models (source of truth)
           │
      FastAPI  →  openapi.json  (generated, checked in)
           │
openapi-typescript  →  packages/contracts/src/generated.ts
           │
    frontend types + thin fetch client
```

Regenerate as a build step and fail CI if the checked-in output is stale, so the
frontend can never compile against a schema the server doesn't actually serve.
That is strictly stronger than two hand-maintained definitions agreeing by
discipline.

| Field               | Where            | Purpose                                                             |
| ------------------- | ---------------- | ------------------------------------------------------------------- |
| `stage`             | summary + result | loading copy, per-stage progress                                    |
| `audio`             | result           | `{ status: 'none'\|'pending'\|'ready'\|'failed', url, durationMs }` |
| `generatedBy`       | result           | `'llm' \| 'template'` — honesty about the fallback                  |
| `weather`           | result           | `{ sprayWindowOk, reason, source }` or null                         |
| `district`          | summary          | which district the advice assumed                                   |
| `source`            | treatment        | citation string from the KB                                         |
| `AudioStatusSchema` | common           | new enum                                                            |

`imageUrl` / `thumbnailUrl` change meaning: now `/api/v1/media/<id>.jpg`, a
cacheable path — no more base64 in JSON. That alone shrinks every response by
hundreds of KB.

---

## 9. Storage

Drop `localStorage`; it was a stand-in and it's now the wrong answer.

- **SQLite** (`aiosqlite`) — one file, no server, real transactions. Table
  `analyses` with the summary columns plus `stage`, `audio_status`, `district`,
  `error_code`. Result JSON in a column until it needs indexing. WAL mode on, so
  reads don't block the job writer.
- **Media on disk** under `apps/api/storage/{images,audio}/`. Served by
  `StaticFiles` (or a thin route) with
  `Cache-Control: public, max-age=31536000, immutable` — content is addressed by
  id and never mutates.
- **Strip EXIF on ingest**, notably GPS. A farmer's photo shouldn't ship their
  coordinates to our disk just because their phone wrote them there. Pillow
  handles this during the downscale, which now happens once, server-side.
- Retention: sweep images older than N days on boot. No unbounded growth.

---

## 10. Failure matrix

Every stage degrades independently. Nothing takes down the analysis.

| Failure                   | Behaviour                          | User sees                   |
| ------------------------- | ---------------------------------- | --------------------------- |
| KB miss for that class    | diagnosis kept, no treatment block | symptom text + "পরামর্শ নিন" |
| Weather timeout           | `weather: null`                    | no weather sentence         |
| LLM unconfigured          | template narrative (`template`)    | same layout, plainer text   |
| LLM error / quota         | template narrative                 | same layout, plainer text   |
| LLM invalid output        | one retry, then template           | as above                    |
| TTS error                 | `audio.status: 'failed'`           | text intact, no player      |
| TTS names a lost medicine | refuse audio, cannot ship it       | text intact, no player      |
| Whole job crash           | `status: failed`, `error_code`     | retry button                |

The rule: **the diagnosis and the curated treatment are never withheld because a
downstream convenience failed.**

---

## 11. Safety rules (non-negotiable)

1. Chemical names and doses come only from the curated KB. A validator checks
   the LLM output for any chemical token not in the facts and rejects on match.
2. `needsProfessionalVerification` is true whenever confidence is below the high
   band, and the UI keeps showing it. Pesticide advice from a 12 MB model is a
   suggestion, not a prescription.
3. Every result carries `isAiEstimate: true` and a citation from the KB.
4. No image or location leaves the box except the text sent to the LLM API —
   and the text contains no image, no GPS, no phone number. District is
   deliberately coarse.
5. Audio is never autoplayed. It's advice, not an alarm.

---

## 12. Configuration

Read with **`pydantic-settings`** from `apps/api/.env`. A missing
`LLM_API_KEY` does _not_ fail startup: the app runs on the template writer, and
`/health` reports `llm: unconfigured` so the degraded state is visible rather
than silent.

```bash
# ---- core ----
APP_ENV=development
PORT=4000
CORS_ORIGIN=http://localhost:5173

# ---- LLM: Tokens BD (optional) ----
LLM_BASE_URL=https://tokens.bd/v1
LLM_API_KEY=                          # leave empty to use the template writer
LLM_MODEL=openai/gpt-6-luna
LLM_TIMEOUT_MS=15000
LLM_MAX_CONCURRENCY=3                 # provider caps at 3 without a plan

# ---- vision ----
VISION_ENABLED=true
VISION_MODEL_DIR=./models             # default checkpoint is committed here
VISION_CONFIDENCE_THRESHOLD=0.6       # raw softmax, before any crop filtering

# ---- TTS ----
TTS_ENABLED=true
TTS_MODEL_ID=sulabhkatiyar/indic-tiny-tts-bengali
TTS_VOICE=female
TTS_FORMAT=opus
FFMPEG_PATH=ffmpeg

# ---- weather ----
OPEN_METEO_BASE_URL=https://api.open-meteo.com/v1
WEATHER_CACHE_MINUTES=30

# ---- storage ----
DATABASE_PATH=./storage/foshol-lens.db
STORAGE_DIR=./storage
PUBLIC_BASE_URL=http://localhost:4000
ANALYSIS_RETENTION_DAYS=30
```

Gone from the old file: `GEMINI_API_KEY`, `GEMINI_MODEL`, `INFERENCE_BASE_URL`.
`VISION_*` keys are back, because vision now exists; they configure a local ONNX
graph and its refusal threshold, not a remote service.

`/health` reports each dependency separately so a failure is diagnosed in one
request:

```json
{
  "status": "degraded",
  "llm": "unconfigured",
  "tts": "ready",
  "weather": "ready",
  "db": "ok",
  "version": "0.1.0"
}
```

`status` is `degraded` whenever the LLM is unconfigured, and the response stays
`200`: the service is up and serving real advice. Do not let a missing LLM key
make the service look healthy — that is the one that fails silently into the
template path.

---

## 13. Frontend changes (done)

Small, because the mock interface was deliberately shaped like the real one.

1. `mock-api.ts` and `api-context.ts` are deleted. `lib/api-client.ts` talks HTTP
   and `mediaUrl()` resolves relative API paths against the API origin.
2. `waitForResult()` polls and reports each `stage` to `AnalysisLoadingState`.
3. `AudioPlayer` appears when `audio.status === 'ready'`, polls while `pending`,
   uses `preload="none"`, and shows the duration.
4. District picker in the upload card, persisted locally, optional.
5. Every visible number is rendered through `toBengaliDigits()`, so `97%` reads
   `৯৭%` and a duration reads `০:৩১`.

6. Keep the uncertainty path as prominent as it is now. It's the honest part.

---

## 14. Phasing

| Phase | Deliverable                                               | Status      |
| ----- | --------------------------------------------------------- | ----------- |
| **0** | SQLite + storage + jobs + template Bangla text            | **Done**    |
| **1** | Curated KB + LLM narrative + validator + fallback         | **Done**    |
| **2** | District weather + deterministic spray window             | **Done**    |
| **3** | TTS + audio subresource + cache + Opus transcode          | **Done**    |
| **4** | ONNX leaf classifier behind a refusal-first decision rule | **Done**    |
| **5** | Confidence calibration + field-photo evaluation           | Not started |

Phase 0 was the load-bearing one. If a photo can't go upload → stored →
rendered with treatment and speech without the LLM, nothing above it is
trustworthy — and it does, on the template path, with no key configured.

---

## 15. Decisions — resolved and open

### Resolved by investigation

| #   | Decision          | Outcome                                                                                                                                   |
| --- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Hardware          | Ryzen 7 5700U, 16 threads, ~3 GiB RAM free, **no GPU acceleration**. CPU-only, memory-tight.                                              |
| 2   | TTS engine        | **`sulabhkatiyar/indic-tiny-tts-bengali`** — measured 9–12× real time, 298 MB peak, 22.5 M params. XTTS-v2 is unusable here (needs CUDA). |
| 3   | Weather           | **Open-Meteo free tier**, verified live. No key, 10k calls/day, 69 ms. Deterministic spray-window rule; LLM only phrases it.              |
| 4   | LLM provider      | **Tokens BD**, `https://tokens.bd/v1`, OpenAI-compatible. Forced tool call for structured output.                                         |
| 5   | Runtime           | **Python 3.12 + FastAPI**, pinned with `uv`. One process.                                                                                 |
| 6   | Vision checkpoint | Deferred by you — does not block phases 0–3.                                                                                              |

### Still open

1. **Vision checkpoint** — the only unimplemented stage. `DEFAULT_DISEASE` holds
   its place; see §4 for the design and the domain-shift problem.
2. **Contract generation** — `packages/contracts` is currently a hand-kept zod
   mirror of `apps/api/app/contracts.py`. Wiring `openapi.json` → TypeScript
   removes the drift risk.
3. **Voice** — `female` is the default. Cheap to switch.
4. **Open-Meteo licensing** — the free tier is not licensed for commercial use.
   Fine for a hackathon; must be replaced or paid for before this ships to
   farmers.

### What was actually done

1. `uv` venv on 3.12, `apps/api` scaffolded with FastAPI + `aiosqlite`.
2. `POST /analyses` stores the image (EXIF stripped), returns 202;
   `pipeline.spawn()` advances `stage` in a detached task.
3. Curated KB: bread-and-butter diseases across rice/potato, each chemical with
   a Bangla name, dose, note, and source. Scope is rice and potato only — the
   two crops the shipped checkpoint can name. Tomato and chili were dropped.
4. `GET /analyses/:id` polls; `GET /analyses/:id/result` serves the existing
   `AnalysisResult` shape; `/image` and `/audio` are separate subresources.
5. Frontend switched from mock to HTTP with no layout change.
6. 39 pytest cases cover the pipeline, the narrative validator, the spray
   verdict, and the two TTS normaliser defects.
