import json
import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from database import get_connection, init_db

AI_ENABLED = bool(os.getenv("GEMINI_API_KEY"))


def make_embedding(text):
    if not AI_ENABLED:
        return None
    from ai_search import make_embedding as _fn
    return _fn(text)


def make_image_embedding(path):
    if not AI_ENABLED:
        return None
    from ai_search import make_image_embedding as _fn
    return _fn(path)


def search_media(query, rows):
    if not AI_ENABLED:
        term = query.casefold()
        return [r for r in rows if term in r["filename"].casefold()]
    from ai_search import search_media as _fn
    return _fn(query, rows)
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("MEDIAVAULT_UPLOAD_DIR", BASE_DIR / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_MB", "50")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm", ".avi"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
print("main.py imported OK", flush=True)
app = FastAPI(title="MediaVault API", version="1.1.0", description="FastAPI + SQLite backend with AI media search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if x.strip()],
    allow_origin_regex=r"https://mediavault-frontend.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

@app.on_event("startup")
def startup():
    init_db()


def public_item(row):
    item = dict(row)
    item.pop("embedding", None)
    item["url"] = f"/uploads/{item['stored_filename']}"
    return item


def read_media_rows():
    with get_connection() as conn:
        return conn.execute("SELECT * FROM media ORDER BY created_at DESC, id DESC").fetchall()


def save_upload(file_path: Path, original_name: str, content_type: str, data: bytes, media_type: str):
    description = original_name
    embedding = None

    try:
        if media_type == "Image":
            vector = make_image_embedding(file_path)
        else:
            vector = make_embedding(original_name)

        if vector:
            embedding = json.dumps(vector)
    except Exception as exc:
        print(f"Embedding generation failed: {exc}")

    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO media
            (filename, stored_filename, media_type, mime_type, size_bytes, description, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                original_name,
                file_path.name,
                media_type,
                content_type,
                len(data),
                description,
                embedding,
            ),
        )
        row = conn.execute(
            "SELECT * FROM media WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    return row
@app.get("/")
def root():
    return {"project": "MediaVault", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "ai_configured": True, "ai_provider": "local CLIP"}
@app.post("/media", status_code=201)
async def upload_media(file: UploadFile = File(...)):
    original_name = Path(file.filename or "upload").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type.")
    content_type = file.content_type or "application/octet-stream"
    if extension in IMAGE_EXTENSIONS and not content_type.startswith("image/"):
        raise HTTPException(400, "File extension and content type do not match.")
    if extension not in IMAGE_EXTENSIONS and not content_type.startswith("video/"):
        raise HTTPException(400, "File extension and content type do not match.")
    data = await file.read(MAX_FILE_BYTES + 1)
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit.")
    stored_name = f"{uuid.uuid4().hex}{extension}"
    path = UPLOAD_DIR / stored_name
    path.write_bytes(data)
    media_type = "Image" if extension in IMAGE_EXTENSIONS else "Video"
    try:
        row = save_upload(path, original_name, content_type, data, media_type)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(500, f"Could not save media metadata: {type(exc).__name__}")
    return public_item(row)


@app.get("/media")
def list_media(
    q: Optional[str] = Query(default=None, max_length=200),
    media_type: str = Query(default="All", pattern="^(All|Image|Video)$"),
):
    with get_connection() as conn:
        if media_type == "All":
            rows = conn.execute("SELECT * FROM media ORDER BY created_at DESC, id DESC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM media WHERE media_type = ? ORDER BY created_at DESC, id DESC", (media_type,)).fetchall()
    items = [public_item(row) for row in rows]
    if q:
        term = q.casefold()
        items = [item for item in items if term in item["filename"].casefold() or term in item["description"].casefold()]
    return {"count": len(items), "items": items}


@app.post("/media/reindex")
def reindex_media():
    rows = read_media_rows()
    updated = 0
    failures = []

    for row in rows:
        path = UPLOAD_DIR / row["stored_filename"]

        if not path.exists():
            failures.append(row["id"])
            continue

        try:
            if row["media_type"] == "Image":
                vector = make_image_embedding(path)
            else:
                vector = make_embedding(row["filename"])

            with get_connection() as conn:
                conn.execute(
                    "UPDATE media SET embedding = ? WHERE id = ?",
                    (json.dumps(vector) if vector else None, row["id"]),
                )

            if vector:
                updated += 1
            else:
                failures.append(row["id"])

        except Exception as error:
            print(f"Reindex failed for media {row['id']}: {error}")
            failures.append(row["id"])

    return {"updated": updated, "failed_ids": failures}

@app.get("/media/{media_id}")
def get_media(media_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Media not found.")
    return public_item(row)


@app.delete("/media/{media_id}")
def delete_media(media_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Media not found.")
        conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
    (UPLOAD_DIR / row["stored_filename"]).unlink(missing_ok=True)
    return {"message": "Media deleted", "id": media_id}


@app.get("/search/ai")
def ai_search(q: str = Query(min_length=1, max_length=500)):
    rows = read_media_rows()

    try:
        results = search_media(q.strip(), rows)
    except Exception as exc:
        raise HTTPException(502, f"AI search failed: {type(exc).__name__}")

    return {
        "query": q,
        "count": len(results),
        "items": [public_item(row) for row in results],
    }

@app.post("/search/visual")
async def visual_search(file: UploadFile = File(...)):
    original_name = Path(file.filename or "reference.png").name
    extension = Path(original_name).suffix.lower()

    if extension not in IMAGE_EXTENSIONS:
        raise HTTPException(400, "Upload a PNG, JPG, JPEG, WEBP, or GIF image.")

    content_type = file.content_type or "image/jpeg"

    if not content_type.startswith("image/"):
        raise HTTPException(400, "The reference file must be an image.")

    data = await file.read(MAX_FILE_BYTES + 1)

    if not data:
        raise HTTPException(400, "The reference image is empty.")

    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "Reference image exceeds the file-size limit.")

    temp_path = UPLOAD_DIR / f"query-{uuid.uuid4().hex}{extension}"

    try:
        temp_path.write_bytes(data)
        query_vector = make_image_embedding(temp_path)

        if query_vector is None:
            raise HTTPException(400, "Could not process the reference image.")

        rows = read_media_rows()
        ranked = []

        for row in rows:
            if row["media_type"] != "Image":
                continue

            try:
                vector = json.loads(row["embedding"]) if row["embedding"] else None
            except (TypeError, json.JSONDecodeError):
                vector = None

            if not vector:
                continue

            from ai_search import cosine_similarity
            score = cosine_similarity(query_vector, vector)

            if score > 0:
                item = dict(row)
                item["similarity"] = round(score, 4)
                ranked.append(item)

        ranked.sort(key=lambda item: item["similarity"], reverse=True)

        return {
            "count": len(ranked),
            "items": [public_item(item) for item in ranked],
        }

        except HTTPException:
        raise
    except Exception as exc:
        print(f"Visual search error: {exc!r}", flush=True)
        raise HTTPException(502, f"Visual search failed: {type(exc).__name__}")
    finally:
        temp_path.unlink(missing_ok=True)
