import streamlit as st
import requests
from datetime import datetime

API_URL = "http://127.0.0.1:8000"
def api_get(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def api_upload(file):
    try:
        response = requests.post(
            f"{API_URL}/media",
            files={
                "file": (
                    file.name,
                    file.getvalue(),
                    file.type or "application/octet-stream"
                )
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)
st.set_page_config(
    page_title="MediaVault | Multimedia Management",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "media" not in st.session_state:
    st.session_state.media = []

if "upload_ids" not in st.session_state:
    st.session_state.upload_ids = set()


# ---------------- DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

.stApp {
    background: #050b27;
    color: #f7f9ff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding: 1.2rem 4% 3rem;
}

h1, h2, h3 {
    color: #ffffff !important;
    letter-spacing: -0.7px;
}

p, label, .stCaption {
    color: #b9c7e8;
}

.nav-brand {
    font-size: 27px;
    font-weight: 900;
    color: white;
    padding-top: 5px;
}

.nav-brand span {
    color: #43b5ff;
}

.brand-caption {
    font-size: 10px;
    letter-spacing: 2px;
    color: #91a9dc;
    font-weight: 700;
}

.nav-line {
    height: 1px;
    background: #243765;
    margin: 15px 0 25px;
}

.hero {
    background:
        radial-gradient(ellipse at 80% 30%,
        #153d80 0%, transparent 48%),
        linear-gradient(120deg, #0b1945, #07102e 75%);
    border: 1px solid #28477e;
    border-radius: 23px;
    padding: 34px;
    margin: 10px 0 28px;
    position: relative;
    overflow: hidden;
}

.eyebrow {
    color: #57d9c5;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.hero-title {
    color: white;
    font-size: clamp(32px, 4vw, 53px);
    font-weight: 900;
    line-height: 1.12;
    letter-spacing: -2px;
}

.hero-title span {
    color: #48b9ff;
}

.hero-copy {
    color: #bdcceb;
    font-size: 15px;
    line-height: 1.8;
    max-width: 650px;
    margin-top: 16px;
}

.hero-art {
    font-size: 110px;
    text-align: center;
    filter: drop-shadow(0 0 25px #268aff55);
}

.metric {
    padding: 20px;
    border-radius: 17px;
    min-height: 142px;
    background: linear-gradient(145deg, #172344, #0c1430);
    border: 1px solid #30416a;
    transition: .2s ease;
}

.metric:hover {
    transform: translateY(-3px);
    border-color: #48b9ff;
}

.metric-label {
    color: #b9c7e8;
    font-size: 13px;
    font-weight: 600;
}

.metric-number {
    font-size: 33px;
    font-weight: 900;
    color: white;
    margin: 12px 0 4px;
}

.metric-note {
    font-size: 11px;
    color: #8298c6;
}

.section-sub {
    color: #aabce1;
    font-size: 13px;
    margin-top: -8px;
    margin-bottom: 20px;
}

.panel {
    background: #0b1430;
    border: 1px solid #26385f;
    border-radius: 17px;
    padding: 22px;
}

.feature-card {
    background: linear-gradient(145deg, #111f43, #0a122c);
    border: 1px solid #2a3d68;
    border-radius: 16px;
    padding: 21px;
    min-height: 185px;
}

.feature-icon {
    font-size: 27px;
    margin-bottom: 12px;
}

.feature-name {
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 10px;
    color: white;
}

.feature-text {
    color: #aabce1;
    font-size: 13px;
    line-height: 1.7;
}

.stButton > button {
    border-radius: 12px;
    min-height: 44px;
    font-weight: 700;
    transition: all .2s ease;
}

.stButton > button:hover {
    border-color: #4dbaff;
    color: #65c6ff;
    transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #158bd9, #4269ed);
    border: none;
    color: white;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 20px #278bff40;
    color: white;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #0b1532;
    border: 1px solid #344873;
    border-radius: 11px;
}

input, textarea {
    color: white !important;
}

[data-testid="stFileUploader"] {
    background: #0b1532;
    border: 1px dashed #42639b;
    border-radius: 15px;
    padding: 12px;
}

.footer {
    color: #8498c2;
    border-top: 1px solid #26385f;
    margin-top: 40px;
    padding-top: 18px;
    font-size: 12px;
}

@media (max-width: 700px) {
    .block-container {
        padding-left: 4%;
        padding-right: 4%;
    }
    .hero {
        padding: 23px;
    }
    .hero-title {
        font-size: 34px;
    }
    .hero-art {
        font-size: 65px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- NAVIGATION ----------------
n1, n2, n3, n4, n5, n6 = st.columns([2.1, 1, 1.35, 1.35, 1.1, 0.8])

with n1:
    st.markdown("""
    <div class="nav-brand">◈ Media<span>Vault</span></div>
    <div class="brand-caption">MULTIMEDIA DATABASE SYSTEM</div>
    """, unsafe_allow_html=True)

pages = [
    ("Home", n2),
    ("Media Library", n3),
    ("Upload Media", n4),
    ("AI Search", n5),
    ("About", n6),
]

for page_name, column in pages:
    with column:
        if st.button(
            page_name,
            key="nav_" + page_name,
            type="primary" if st.session_state.page == page_name else "secondary",
            use_container_width=True
        ):
            st.session_state.page = page_name
            st.rerun()

st.markdown('<div class="nav-line"></div>', unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def go_to(page):
    st.session_state.page = page
    st.rerun()


def get_stats():
    media = st.session_state.media
    images = sum(x["type"] == "Image" for x in media)
    videos = sum(x["type"] == "Video" for x in media)
    storage = sum(x["size"] for x in media)
    return len(media), images, videos, storage


def metric_card(icon, label, number, note):
    st.markdown(f"""
    <div class="metric">
        <div style="font-size:23px">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-number">{number}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- HOME / DASHBOARD ----------------
if st.session_state.page == "Home":

    total, images, videos, storage = get_stats()

    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        st.markdown(f"""
        <div class="hero">
            <div class="eyebrow">YOUR DIGITAL ASSET HUB</div>
            <div class="hero-title">
                Your Media.<br>
                <span>Organized Smarter.</span>
            </div>
            <div class="hero-copy">
                Upload, discover, preview and organize your images
                and videos in one modern multimedia workspace.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explore Media  →", type="primary"):
            go_to("Media Library")

        if st.button("＋ Upload Your Files"):
            go_to("Upload Media")

    with right:
        st.markdown("""
        <div class="hero" style="height:300px;display:flex;
             flex-direction:column;justify-content:center;align-items:center;">
            <div class="hero-art">▣</div>
            <div style="font-size:25px;font-weight:800;text-align:center;">
                MediaVault
            </div>
            <div class="eyebrow" style="margin-top:12px;">
                STORE · DISCOVER · ORGANIZE
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Dashboard Overview")
    st.markdown(
        '<div class="section-sub">A quick snapshot of your media collection.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        metric_card("🗂️", "Total Assets", total, "Files in this session")
    with c2:
        metric_card("🖼️", "Images", images, "Image assets")
    with c3:
        metric_card("🎬", "Videos", videos, "Video assets")
    with c4:
        metric_card("💾", "Storage Used",
                    f"{storage / 1024:.1f} KB", "Session storage")

    st.markdown("## Your Workspace")

    f1, f2, f3 = st.columns(3, gap="medium")

    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔎</div>
            <div class="feature-name">Explore Your Media</div>
            <div class="feature-text">
                Search filenames and filter images or videos.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Library →", key="home_lib", use_container_width=True):
            go_to("Media Library")

    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📤</div>
            <div class="feature-name">Upload Center</div>
            <div class="feature-text">
                Add images and videos and preview them before saving.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Upload Media →", key="home_upload", use_container_width=True):
            go_to("Upload Media")

    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">ℹ️</div>
            <div class="feature-name">About the Project</div>
            <div class="feature-text">
                Learn about the system, its features and technologies.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Project Details →", key="home_about", use_container_width=True):
            go_to("About")


# ---------------- MEDIA LIBRARY ----------------
elif st.session_state.page == "Media Library":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">YOUR DIGITAL COLLECTION</div>
        <div class="hero-title">Explore Your <span>Media.</span></div>
        <div class="hero-copy">
            Browse and search images and videos stored in your MediaVault backend.
        </div>
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input(
        "Search media",
        placeholder="Search by filename..."
    )

    filter_type = st.selectbox(
        "Media type",
        ["All Files", "Images", "Videos"]
    )

    type_map = {
        "All Files": "All",
        "Images": "Image",
        "Videos": "Video"
    }

    result, error = api_get(
        "/media",
        params={"q": search, "media_type": type_map[filter_type]}
    )

    if error:
        st.error(f"Could not load media: {error}")
    else:
        items = result.get("items", [])
        st.caption(f"{len(items)} file(s) found")

        if not items:
            st.info("No files found. Upload some media to get started.")
        else:
            cols = st.columns(3, gap="medium")

            for i, item in enumerate(items):
                with cols[i % 3]:
                    with st.container(border=True):
                        media_url = f"{API_URL}{item['url']}"

                        if item["media_type"] == "Image":
                            st.image(media_url, use_container_width=True)
                        else:
                            st.video(media_url)

                        st.markdown(f"**{item['filename']}**")
                        st.caption(item["media_type"])
                        st.caption(
                            f"{item['size_bytes'] / 1024:.1f} KB"
                        )

                        if st.button(
                            "Delete",
                            key=f"delete_{item['id']}"
                        ):
                            try:
                                response = requests.delete(
                                    f"{API_URL}/media/{item['id']}",
                                    timeout=30
                                )
                                response.raise_for_status()
                                st.success("Media deleted.")
                                st.rerun()
                            except requests.RequestException as exc:
                                st.error(f"Delete failed: {exc}")

# ---------------- UPLOAD MEDIA ----------------
elif st.session_state.page == "Upload Media":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">UPLOAD CENTER</div>
        <div class="hero-title">Add Your <span>Media.</span></div>
        <div class="hero-copy">
            Upload images and videos to your persistent MediaVault library.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose images or videos",
        type=["png", "jpg", "jpeg", "webp", "gif",
              "mp4", "mov", "webm", "avi"],
        accept_multiple_files=True,
        key="backend_media_uploader"
    )

    if uploaded_files:
        st.markdown("### Preview Selected Files")

        for file in uploaded_files:
            with st.container(border=True):
                if file.type.startswith("image/"):
                    st.image(file, use_container_width=True)
                elif file.type.startswith("video/"):
                    st.video(file)

                st.markdown(f"**{file.name}**")
                st.caption(f"Size: {len(file.getvalue()) / 1024:.1f} KB")

        if st.button("Save Files to MediaVault →", type="primary"):
            successful = 0

            for file in uploaded_files:
                with st.spinner(f"Uploading {file.name}..."):
                    result, error = api_upload(file)

                if error:
                    st.error(f"Upload failed for {file.name}: {error}")
                else:
                    successful += 1

            if successful:
                st.success(f"{successful} file(s) saved successfully!")
                st.session_state.page = "Media Library"
                st.rerun()

    st.markdown("""
    <div class="panel">
        <b>Supported formats</b><br><br>
        Images: PNG, JPG, JPEG, WEBP, GIF<br>
        Videos: MP4, MOV, WEBM, AVI
    </div>
    """, unsafe_allow_html=True)

# ---------------- AI SEARCH ----------------
elif st.session_state.page == "AI Search":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">POWERED BY LOCAL CLIP</div>
        <div class="hero-title">Find Media <span>With AI.</span></div>
        <div class="hero-copy">
            Describe what you're looking for in natural language.
        </div>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Describe the media you're looking for",
        placeholder="e.g. a cat, a red car, a person outdoors..."
    )

    if st.button("Search with AI", type="primary"):
        if not query.strip():
            st.warning("Please enter a search description.")
        else:
            with st.spinner("Searching your media library..."):
                result, error = api_get(
                    "/search/ai",
                    params={"q": query.strip()}
                )

            if error:
                st.error(f"AI search failed: {error}")
            else:
                items = result.get("items", [])
                st.success(f"Found {len(items)} matching file(s).")

                if not items:
                    st.info(
                        "No matching results. Upload media first and "
                        "make sure the files have embeddings."
                    )
                else:
                    cols = st.columns(3, gap="medium")

                    for i, item in enumerate(items):
                        with cols[i % 3]:
                            with st.container(border=True):
                                media_url = f"{API_URL}{item['url']}"

                                if item["media_type"] == "Image":
                                    st.image(
                                        media_url,
                                        use_container_width=True
                                    )
                                else:
                                    st.video(media_url)

                                st.markdown(f"**{item['filename']}**")
                                st.caption(item["media_type"])

                                if "similarity" in item:
                                    st.caption(
                                        f"Similarity: "
                                        f"{item['similarity']:.2%}"
                                    )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    ◈ MediaVault · Multimedia Database Management System
    <span style="float:right">Store · Discover · Organize</span>
</div>
""", unsafe_allow_html=True)
