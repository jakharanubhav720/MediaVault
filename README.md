# MediaVault — Full Streamlit + FastAPI + SQLite Project

## Features
- Streamlit frontend retaining the MediaVault design.
- FastAPI backend for upload, list, filter, search, and delete.
- SQLite stores media metadata; files are saved in `uploads/`.
- Optional AI-generated image captions and embeddings.
- Lightweight visual similarity search and optional Gemini-assisted descriptions/search.
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
- Render Free uses the lightweight `visual-v1` image descriptor for reference-image similarity; it does not load CLIP or FastEmbed.
- Natural-language search uses optional Gemini text embeddings when `GEMINI_API_KEY` is configured, with filename/description keyword matching as a fallback.
- Optional Gemini image captions remain available only when `GEMINI_API_KEY` is configured.
- To enable CLIP on a larger host, install `fastembed`, set both `IMAGE_EMBED_BACKEND=clip` and `ENABLE_CLIP=true`, deploy, and re-index existing photos.
- After changing image embedding backends, call `POST /media/reindex` once to rebuild compatible embeddings.
- Video files are not currently analyzed frame-by-frame; they can still be found by filename/description. Video-frame extraction is a future enhancement.
- Similarity values are ranking scores, not calibrated probabilities that two images are identical.

### Render Free deployment
- Keep `ENABLE_CLIP` unset or set it to `false`; CLIP is disabled by default to stay within Render Free's 512 MB memory limit.
- Do not add `fastembed` to the Render Free requirements.
- Uploads, the media library, deletion, lightweight visual similarity, optional Gemini captions, and keyword search remain available.
- Re-index saved images after this backend deploy if you want all existing photos to use the lightweight visual descriptor.

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
