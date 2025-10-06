# Multimodal Linear Regression Oral Exam Platform

This repository contains a FastAPI-based backend that administers an oral assessment on linear regression. It orchestrates audio transcription, retrieval-augmented grading, and multimodal delivery analysis in near real time. The system is intentionally modular so it can run entirely on free/open models while still allowing upgrades to hosted APIs when desired.

## Features

- **Question retrieval** – Questions and curated reference answers are stored in a Chroma vector database. Each session randomly selects a prompt from this bank and persists it for later grading.
- **Speech-to-text** – Audio submissions are transcribed with [OpenAI Whisper](https://github.com/openai/whisper) (any local size can be configured).
- **Retrieval-Augmented Evaluation** – The transcript is matched against vetted context chunks and graded by either a local `llama.cpp` model or a deterministic heuristic. Output is a structured JSON summary of strengths, missed concepts, and accuracy scores.
- **Computer vision & VLM analysis** – Key frames are sampled from the webcam video, gestures are detected with MediaPipe, and optional descriptions can be produced with an open VLM such as LLaVA.
- **Composite reporting** – Content and delivery insights are fused into a final instructor-facing report with configurable weighting.
- **Asynchronous processing** – Long-running workloads can be dispatched either via FastAPI background tasks or the included Celery worker template for horizontal scaling.

## Project Layout

```
app/
  api/            # REST endpoints
  config.py       # Pydantic settings (env aware)
  db/             # Vector store wrapper (Chroma)
  models/         # Dataclasses for domain objects
  services/       # Core orchestration: STT, RAG, CV/VLM, reporting, etc.
  main.py         # FastAPI application factory
celery_app.py     # Celery worker entry point (optional)
data/questions.json
requirements.txt
```

## Quick Start

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   > ℹ️ Whisper and MediaPipe rely on `ffmpeg`. Install it through your OS package manager when running the full pipeline.

2. **Prepare optional models**

   - **LLM (content grading)**: download any `gguf` model compatible with [`llama.cpp`](https://github.com/ggerganov/llama.cpp). Update `INTERVIEW_LLM_MODEL_PATH` in `.env` to point to the file. If omitted, the system falls back to a deterministic heuristic.
   - **Vision-Language Model**: to enable multimodal narration, set `INTERVIEW_ENABLE_VLM=true` and `INTERVIEW_VLM_MODEL_NAME` to a Hugging Face repo such as `liuhaotian/llava-v1.5-7b-hf`. Running these models typically requires a GPU.

3. **Start the API**

   ```bash
   uvicorn app.main:app --reload
   ```

4. **(Optional) Start Celery worker**

   ```bash
   celery -A celery_app.celery_app worker --loglevel=info
   ```

   Set `INTERVIEW_USE_CELERY=true` in your environment and the API will enqueue jobs on the Celery worker instead of processing them inline. The built-in FastAPI background tasks remain the default for lightweight experimentation.

## API Walkthrough

1. `POST /exam/start` – Creates a new session, selects a question, and returns `{session_id, question, max_response_duration}`.
2. `POST /exam/{session_id}/submit` – Accepts multipart form data containing optional audio (`audio`), optional video (`video`), and/or a text transcript (`transcript`). The submission is queued for processing.
3. `POST /exam/submit-transcript` – Lightweight JSON endpoint for manual transcripts without media uploads.
4. `GET /exam/{session_id}` – Poll for session status and retrieve the final report once processing finishes.
5. `GET /health` – Simple liveness probe.

The final report is structured as:

```json
{
  "question_asked": "...",
  "student_answer_transcript": "...",
  "content_analysis": {
    "accuracy_score": 82.5,
    "missed_concepts": ["Gauss-Markov optimality"],
    "errors_made": ["Confused homoscedasticity with independence"],
    "correct_points": ["Defined least squares objective"],
    "confidence": 78.0
  },
  "delivery_analysis": {
    "gestures_detected": ["pointing", "counting"],
    "confidence_estimate": "high",
    "engagement_level": "medium",
    "notes": ["Student appeared confident and used demonstrative gestures..."]
  },
  "composite_score": 84.1
}
```

## Configuration Reference

All settings may be overridden via environment variables (prefix `INTERVIEW_`). Key options:

| Variable | Default | Description |
| --- | --- | --- |
| `INTERVIEW_DATASET_PATH` | `data/questions.json` | Location of curated question bank |
| `INTERVIEW_VECTOR_STORE_PATH` | `storage/vector_store` | Persistence directory for Chroma |
| `INTERVIEW_TRANSCRIPTION_MODEL` | `base` | Whisper model variant |
| `INTERVIEW_LLM_MODEL_PATH` | `None` | Path to a local `gguf` model used for grading |
| `INTERVIEW_ENABLE_VLM` | `False` | Enables the VLM layer (requires GPU-capable model) |
| `INTERVIEW_VLM_MODEL_NAME` | `None` | Hugging Face repository for the VLM |
| `INTERVIEW_COMPOSITE_SCORE_CONTENT_WEIGHT` | `0.7` | Weight assigned to content accuracy |
| `INTERVIEW_COMPOSITE_SCORE_DELIVERY_WEIGHT` | `0.3` | Weight assigned to delivery |
| `INTERVIEW_CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `INTERVIEW_CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | Celery results backend |
| `INTERVIEW_USE_CELERY` | `False` | Dispatch processing jobs to Celery instead of FastAPI background tasks |

## Extending the System

- **Additional questions**: append entries to `data/questions.json` (ensure unique `id`s). They are automatically embedded into the Chroma collection on startup.
- **Alternate STT or VLM backends**: implement a new service satisfying the respective interfaces (`TranscriptionService` or `BaseVisionLanguageModel`) and wire it through `app/dependencies.py`.
- **Analytics dashboards**: consume the JSON reports produced by `AssessmentPipeline` to visualise student progress or feed downstream grading workflows.

## Testing Notes

Due to the heavy multimedia dependencies, unit tests are not bundled with this prototype. When integrating into CI, prefer mocking the Whisper, LLM, and MediaPipe layers to avoid large downloads while still exercising the orchestration code paths.

## Air Interaction Demo Tool

`air_interact.py` is a standalone real-time "air-interaction" capture utility that accompanies the oral exam platform. It tracks the user's index fingertip, segments pen-up/pen-down strokes, recognises air-drawn shapes and dynamic gestures, and streams JSON events for downstream processing.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python mediapipe numpy
```

### Runtime Modes

| Mode | Description |
| --- | --- |
| `--mode dl` | MediaPipe Hands for landmark detection with pinch-based pen gating. |
| `--mode cv` | OpenCV-only pipeline using motion/contour analysis (no neural nets). |

### Launch

```bash
python air_interact.py --mode dl --camera 0 --record-start off --face box --log-file session_log.jsonl
```

The OpenCV window is titled **Air Interaction**. Ensure the window has focus so hotkeys are received.

### Hotkeys

| Key | Action |
| --- | --- |
| `R` | Toggle recording (REC: ON/OFF). |
| `F` | Cycle face overlay (off → box → picture-in-picture). |
| `C` | Clear the on-screen canvas and stroke buffer. |
| `T` | Save the last stroke as a new $1 template to refine symbol recognition. |
| `ESC` | Quit. |

### Recognised Acts

| Category | Labels / Gestures |
| --- | --- |
| Shapes | line, circle, square, rectangle, triangle, check ✓, cross X, plus +, minus −, arrow →, zigzag, dot |
| Fingers | Numbers 0–10 (detected from one or two hands depending on the mode) |
| Dynamic | pinch (pen-down), unpinch (pen-up), erase swipe |

### JSON Event Stream

Events are emitted to `stdout` and optionally to `--log-file` as JSON lines with both epoch and ISO-8601 timestamps. Examples include `session_start`, `recording`, `pen_down`, `stroke_end`, `gesture`, and `template_saved`.

