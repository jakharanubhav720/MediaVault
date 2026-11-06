# MediaVault Backend — Local CLIP Edition

FastAPI + SQLite backend for MediaVault. This `local-clip` branch enables local CLIP semantic text-to-image and image-to-image search.

## Requirements

- Python 3.10 or newer
- A machine with at least 4 GB RAM (8 GB recommended)
- Several GB of free disk space for downloaded model files
- macOS Apple Silicon or a compatible Linux/Windows Python environment

## Setup (macOS)

Run from the backend repository folder:

```bash
git checkout local-clip
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in this folder:

```env
IMAGE_EMBED_BACKEND=clip
ENABLE_CLIP=true
GEMINI_API_KEY=
```

CLIP does not require a Gemini API key. Gemini is optional for image captions and supplementary text embeddings. Keep `.env` private.

## Start the backend

```bash
source venv/bin/activate
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The first image upload or CLIP search downloads the CLIP models. Allow the initial download to finish. New image uploads are embedded with CLIP automatically.

## Existing media

After switching from the lightweight visual descriptor to CLIP, re-index existing images once:

```bash
curl -X POST http://127.0.0.1:8000/media/reindex
```

Or use the Re-index saved images button in the frontend.

## Frontend connection

When running the Next.js frontend locally, set its API base URL to:

```env
NEXT_PUBLIC_MEDIAVAULT_API_URL=http://127.0.0.1:8000
```

Restart the Next.js dev server after changing this variable. The Vercel-hosted frontend cannot reach a backend bound only to localhost; run the frontend locally too, or expose the local backend securely using a tunnel.

## CLIP behavior

- CLIP text and image encoders share a vector space, enabling natural-language-to-image search and reference-image similarity.
- Similarity percentages are ranking indicators, not calibrated probabilities.
- Video files are not currently embedded frame-by-frame; they remain searchable by filename/description.
- The CLIP model is loaded lazily when first needed, rather than at API startup.

## Render Free safety

Do not deploy this `local-clip` branch to the Render Free 512 MB instance. Keep Render on `main`, where CLIP is disabled and FastEmbed is not installed.

## API routes

- `GET /health`
- `POST /media` multipart field `file`
- `GET /media?q=term&media_type=All|Image|Video`
- `GET /media/{id}`
- `DELETE /media/{id}`
- `POST /media/reindex`
- `GET /search/ai?q=describe%20an%20image`
- `POST /search/visual` multipart field `file`
