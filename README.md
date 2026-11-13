# MediaVault Backend — Local SigLIP 2 Edition

FastAPI + SQLite backend for MediaVault. This `local-clip` branch now uses Google's SigLIP 2 Base for local semantic text-to-image and image-to-image search. The branch name is retained so existing local clones do not need to be recloned.

## Requirements

- Python 3.10 or newer
- Apple Silicon Mac or a compatible PyTorch machine
- 16 GB RAM recommended for this local setup
- Several GB of free disk space for model weights and tokenizer files

## Setup (macOS)

Run from the backend repository folder:

```bash
git checkout local-clip
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create or update `.env` in this folder:

```env
IMAGE_EMBED_BACKEND=siglip2
ENABLE_SIGLIP2=true
SIGLIP2_MODEL_ID=google/siglip2-base-patch16-256
SIGLIP2_TEXT_MIN_SCORE=0.18
GEMINI_API_KEY=
MEDIAVAULT_UPLOAD_DIR=uploads
MAX_FILE_MB=50
```

The old local setting `IMAGE_EMBED_BACKEND=clip` is temporarily accepted as a migration alias and selects SigLIP 2. No Gemini API key is required for SigLIP 2 embeddings. Gemini remains optional for image captions.

## Start the backend

```bash
source venv/bin/activate
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The first image upload or search downloads the SigLIP 2 model and processor from Hugging Face. This can take several minutes. The model is loaded lazily and reused. On Apple Silicon, the backend uses PyTorch MPS when available and otherwise falls back to CPU.

## Existing media

All existing image embeddings must be regenerated after changing from CLIP or visual-v1 to SigLIP 2. Start the backend, then run:

```bash
curl -X POST http://127.0.0.1:8000/media/reindex
```

Confirm the response reports `"model":"siglip2-base-patch16-256"` and no failed IDs.

## Frontend connection

When running the Next.js frontend locally, set:

```env
NEXT_PUBLIC_MEDIAVAULT_API_URL=http://127.0.0.1:8000
```

Restart the Next.js dev server after changing this variable.

## Search behavior

- SigLIP 2 image and text encoders produce vectors in a shared embedding space.
- Text search and uploaded-reference visual search both use the same SigLIP 2 image vectors.
- `SIGLIP2_TEXT_MIN_SCORE` controls the minimum cosine similarity used for semantic text results. Tune it against a labelled test set; it is not an accuracy percentage.
- Similarity values are ranking signals, not calibrated probabilities.
- Video files are not embedded frame-by-frame; they remain searchable by filename/description.

## Render Free safety

Do not deploy this local-model branch to Render Free's 512 MB instance. Keep the Render deployment on `main`.

## API routes

- `GET /health`
- `POST /media` multipart field `file`
- `GET /media?q=term&media_type=All|Image|Video`
- `GET /media/{id}`
- `DELETE /media/{id}`
- `POST /media/reindex`
- `GET /search/ai?q=describe%20an%20image`
- `POST /search/visual` multipart field `file`


## Smart organization features

This branch also includes three local organization features:

- **Automatic AI tags:** image uploads are compared with a small category vocabulary using the existing SigLIP 2 image/text embedding space. Tags are stored as JSON in SQLite. Existing images can be tagged again with `POST /media/reindex`.
- **Duplicate finder:** `POST /duplicates/scan` refreshes SHA-256 and perceptual difference-hash fingerprints; `GET /duplicates` returns exact and visually similar groups. Similarity groups are suggestions only—files are never deleted automatically.
- **Smart albums:** `GET /albums` groups tagged images into Food, Travel, Nature, People, Pets & Animals, Vehicles, Documents, Architecture, Events and Technology. `GET /albums/{slug}` returns one album.

SQLite startup migrations add the `tags`, `sha256` and `phash` columns without deleting existing media. The Next.js `/organize` page provides the Smart Albums and Duplicate Finder interface. Re-index existing images after pulling this branch to populate tags and fingerprints.
