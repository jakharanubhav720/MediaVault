# MediaVault — Full Streamlit + FastAPI + SQLite Project

## Features
- Streamlit frontend retaining the MediaVault design.
- FastAPI backend for upload, list, filter, search, and delete.
- SQLite stores media metadata; files are saved in `uploads/`.
- Optional AI-generated image captions and embeddings.
- Semantic text search and reference-image search.
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
- Uploading an image generates a caption and text embedding when the API key is configured.
- Text search embeds a natural-language query and ranks stored media by cosine similarity.
- Reference-image search captions the reference image and compares its meaning with stored media embeddings. It does not currently use a dedicated CLIP image encoder.
- Video files are currently indexed using filenames; video-frame extraction is a future enhancement.
- Existing media can be indexed using `POST /media/reindex` after setting the key.

AI provider calls may cost money. Check current API pricing and usage limits.

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
