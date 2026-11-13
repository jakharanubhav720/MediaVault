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

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("MEDIAVAULT_UPLOAD_DIR", BASE_DIR / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_MB", "50")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm", ".avi"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
print("main.py imported OK", flush=True)


def ai():
    """Import the AI helpers lazily so the web server always starts quickly."""
    import ai_search
    return ai_search


app = FastAPI(title="MediaVault API", version="1.2.0", description="FastAPI + SQLite backend with image similarity search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.on_event("startup")
def startup():
    init_db()


def public_item(row):
    item = dict(row)
    item.pop("embedding", None)  # similarity / match_percent are intentionally kept
    item["url"] = f"/uploads/{item['stored_filename']}"
    return item


def read_media_rows():
    with get_connection() as conn:
        return conn.execute("SELECT * FROM media ORDER BY created_at DESC, id DESC").fetchall()


def save_upload(file_path: Path, original_name: str, content_type: str, data: bytes, media_type: str):
    description = original_name
    embedding = None
    tags = []
    sha256 = None
    phash = None

    try:
        ai_description, record = ai().build_record(file_path, media_type, original_name)
        if ai_description:
            description = ai_description
        embedding = json.dumps(record)
    except Exception as exc:
        print(f"Embedding generation failed: {exc!r}", flush=True)

    if media_type == "Image":
        try:
            from media_features import image_fingerprints, tag_image
            sha256, phash = image_fingerprints(file_path)
            tags = tag_image(file_path)
        except Exception as exc:
            print(f"Automatic tagging/fingerprinting skipped: {exc!r}", flush=True)

    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO media
            (filename, stored_filename, media_type, mime_type, size_bytes, description, embedding, tags, sha256, phash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (original_name, file_path.name, media_type, content_type, len(data),
             description, embedding, json.dumps(tags), sha256, phash),
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
    helper = ai()
    return {
        "status": "ok",
        "image_model": helper.current_image_model(),
        "text_search": helper.current_text_model(),
        "embedding_backend": "local_siglip2" if helper.current_image_model() == "siglip2-base-patch16-256" else "visual",
    }


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
    # Remove stale database entries whose uploaded files no longer exist.
    # This prevents broken image/video cards from returning after a refresh.
    with get_connection() as conn:
        stale_rows = conn.execute("SELECT id, stored_filename FROM media").fetchall()
        stale_ids = [
            row["id"] for row in stale_rows
            if not (UPLOAD_DIR / row["stored_filename"]).is_file()
        ]
        if stale_ids:
            conn.executemany("DELETE FROM media WHERE id = ?", [(media_id,) for media_id in stale_ids])

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
    """Rebuild embeddings for every stored file with the current model."""
    helper = ai()
    rows = read_media_rows()
    updated = 0
    failures = []

    for row in rows:
        path = UPLOAD_DIR / row["stored_filename"]

        if not path.exists():
            failures.append(row["id"])
            continue

        try:
            description, record = helper.build_record(path, row["media_type"], row["filename"])
            with get_connection() as conn:
                if description:
                    conn.execute("UPDATE media SET embedding = ?, description = ? WHERE id = ?",
                                 (json.dumps(record), description, row["id"]))
                else:
                    conn.execute("UPDATE media SET embedding = ? WHERE id = ?", (json.dumps(record), row["id"]))
            updated += 1
        except Exception as error:
            print(f"Reindex failed for media {row['id']}: {error!r}", flush=True)
            failures.append(row["id"])

    return {"updated": updated, "failed_ids": failures, "model": helper.current_image_model()}


@app.post("/duplicates/scan")
def scan_duplicates():
    """Refresh image fingerprints, then return exact and visually similar groups."""
    from media_features import duplicate_groups, image_fingerprints
    rows = read_media_rows()
    indexed, failed = 0, []
    for row in rows:
        if row["media_type"] != "Image":
            continue
        try:
            sha256, phash = image_fingerprints(UPLOAD_DIR / row["stored_filename"])
            with get_connection() as conn:
                conn.execute("UPDATE media SET sha256 = ?, phash = ? WHERE id = ?",
                             (sha256, phash, row["id"]))
            indexed += 1
        except Exception as exc:
            failed.append(row["id"])
            print(f"Duplicate scan failed for {row['id']}: {exc!r}", flush=True)
    groups = duplicate_groups(read_media_rows(), UPLOAD_DIR)
    return {"scanned": indexed, "failed_ids": failed, "group_count": len(groups), "groups": groups}


@app.get("/duplicates")
def get_duplicates():
    from media_features import duplicate_groups
    groups = duplicate_groups(read_media_rows(), UPLOAD_DIR)
    return {"group_count": len(groups), "groups": groups}


@app.get("/albums")
def get_smart_albums():
    """Build smart albums dynamically from saved AI tags."""
    albums = {}
    for row in read_media_rows():
        if row["media_type"] != "Image":
            continue
        try:
            tags = json.loads(row["tags"] or "[]")
        except (TypeError, ValueError, KeyError):
            tags = []
        for tag in tags:
            name = tag.get("album")
            if not name:
                continue
            album = albums.setdefault(name, {"name": name, "slug": name.lower().replace(" & ", "-").replace(" ", "-"), "count": 0, "items": []})
            album["count"] += 1
            item = public_item(row)
            item["tags"] = tags
            album["items"].append(item)
    return {"count": len(albums), "albums": sorted(albums.values(), key=lambda album: album["name"])}


@app.get("/albums/{slug}")
def get_smart_album(slug: str):
    data = get_smart_albums()
    for album in data["albums"]:
        if album["slug"] == slug:
            return album
    raise HTTPException(404, "Smart album not found. Re-index images to generate AI tags.")


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
        results = ai().search_media(q.strip(), rows)
    except Exception as exc:
        print(f"AI search error: {exc!r}", flush=True)
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
        helper = ai()

        try:
            query = helper.image_vector(temp_path)
        except ValueError as exc:
            raise HTTPException(400, f"Could not process the reference image: {exc}")

        ranked = []
        not_indexed = 0

        for row in read_media_rows():
            if row["media_type"] != "Image":
                continue

            entry = helper.load_record(row["embedding"]).get("img")

            # Only compare vectors produced by the same model as the reference image.
            if not entry or entry.get("model") != query["model"] or not entry.get("vec"):
                not_indexed += 1
                continue

            score = helper.cosine_similarity(query["vec"], entry["vec"])
            item = dict(row)
            item["similarity"] = round(score, 4)  # Keep raw cosine for ranking.
            # Calibrate the displayed relevance score: unrelated pairs around 0.54
            # should not look like 54% matches. Exact matches remain 100%.
            relevance_floor = float(os.getenv("VISUAL_MATCH_FLOOR", "0.60"))
            calibrated = (score - relevance_floor) / max(1e-9, 1.0 - relevance_floor)
            item["match_percent"] = round(min(1.0, max(0.0, calibrated)) * 100, 1)
            ranked.append(item)

        ranked.sort(key=lambda item: item["similarity"], reverse=True)

        return {
            "count": len(ranked),
            "model": query["model"],
            "not_indexed": not_indexed,
            "items": [public_item(item) for item in ranked],
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Visual search error: {exc!r}", flush=True)
        raise HTTPException(502, f"Visual search failed: {type(exc).__name__}")
    finally:
        temp_path.unlink(missing_ok=True)
