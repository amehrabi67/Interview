# Multimodal RAG Assessment Agent for Linear Regression

This project implements a FastAPI-based service that simulates an oral examination workflow for linear regression topics. The system orchestrates multimodal capture, transcription, retrieval-augmented evaluation, visual analysis, and report generation.

## Features

- **Question retrieval** using an in-memory vector store seeded with curated linear regression questions and references.
- **Audio/video capture interface** with a mock recorder that can be swapped for real hardware integrations.
- **Speech-to-text services** via an extensible abstraction (defaulting to a text file fallback for local testing).
- **RAG content evaluation** that compares transcripts against authoritative answers and surfaces covered or missed concepts.
- **Visual analysis** that aggregates gesture metadata from key frames to estimate delivery confidence and engagement.
- **Report synthesis** combining content and delivery insights into a composite score.

## Running the API

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```

3. Interact with the endpoints:
   - `GET /question` returns a random question.
   - `POST /assess` accepts a manual transcript and optional gesture annotations and returns a comprehensive JSON report.

## Testing

Execute the automated tests with:

```bash
pytest
```

The tests validate the vector retrieval logic and the end-to-end orchestration pipeline using the mock recorder and fallback transcription.
