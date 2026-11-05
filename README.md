# MediaVault — Full Streamlit + FastAPI + SQLite Project

## Features
- Streamlit frontend retaining the MediaVault design.
- FastAPI backend for upload, list, filter, search, and delete.
- SQLite stores media metadata; files are saved in `uploads/`.
- Optional AI-generated image captions and embeddings.
- Local CLIP semantic search: natural-language-to-image and reference-image-to-image search.
- Reindex existing media after configuring the AI key.

## Setup (macOS)
Extract this folder and run the following commands from inside it:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` if you want AI features. Keep `.env` private.

## Run both services
Open Terminal 1 (backend):
```bash
cd mediavault_backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

Open Terminal 2 (Streamlit frontend):
```bash
cd mediavault_backend
source venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501`. API documentation is at `http://127.0.0.1:8000/docs`.

## AI features
- CLIP runs locally in the FastAPI backend through FastEmbed; no external AI API key is needed for CLIP search.
- New photos receive a 512-dimensional CLIP vision embedding when uploaded.
- Natural-language search embeds the query with CLIP's text encoder and compares it with saved photo embeddings.
- Reference-image search embeds the uploaded reference with CLIP's vision encoder and compares it with saved photo embeddings.
- CLIP's text and image encoders share the same vector space, enabling text-to-image and image-to-image retrieval.
- Existing photos must be re-indexed after switching to CLIP: call `POST /media/reindex` once after deployment, or use the Re-index saved images button in the AI Search page.
- Video files are not currently analyzed frame-by-frame; they can still be found by filename/description. Video-frame extraction is a future enhancement.
- Optional Gemini image captions remain available only when `GEMINI_API_KEY` is configured. CLIP search itself does not call Gemini or another hosted AI API.
- Similarity values are ranking scores, not calibrated probabilities that two images are identical.

### Render deployment
- Add `IMAGE_EMBED_BACKEND=clip` in the Render backend environment (the code defaults to CLIP if the variable is absent).
- Deploy the backend, wait for the first model download, then call `POST /media/reindex` once to rebuild old photo embeddings.
- The first CLIP use downloads the model files to the server; ensure the Render service has enough memory and persistent disk if you need model cache and uploaded media to survive restarts.

## API routes
- `GET /health`
- `POST /media` multipart field `file`
- `GET /media?q=term&media_type=All|Image|Video`
- `GET /media/{id}`
- `DELETE /media/{id}`
- `POST /media/reindex`
- `GET /search/ai?q=describe%20an%20image`
- `POST /search/visual` multipart field `file`

## Notes
This is a college-project starter. It is not production-hardened. Before public deployment add authentication/authorization, private media access, quotas, stronger file-content validation, rate limits, and backups.
