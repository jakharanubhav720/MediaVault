"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";

type Media = {
  id: number; filename: string; media_type: string; size_bytes?: number;
  created_at?: string; description?: string; url?: string; similarity?: number; match_percent?: number;
};
const API = (process.env.NEXT_PUBLIC_MEDIAVAULT_API_URL || "https://mediavault-mpp4.onrender.com").replace(/\/$/, "");
const pages = ["Home", "Media Library", "Upload Media", "AI Search", "About"] as const;
type Page = typeof pages[number];

export default function Home() {
  const [page, setPage] = useState<Page>("Home");
  const [items, setItems] = useState<Media[]>([]);
  const [status, setStatus] = useState("");
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");
  const [aiQuery, setAiQuery] = useState("");
  const [aiResults, setAiResults] = useState<Media[]>([]);
  const [visualFile, setVisualFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);

  async function loadMedia() {
    try {
      const r = await fetch(`${API}/media`, { cache: "no-store" });
      if (!r.ok) throw new Error(`API returned ${r.status}`);
      const data = await r.json();
      setItems(Array.isArray(data.items) ? data.items : []);
      setApiOk(true);
    } catch (e) {
      setApiOk(false);
      setStatus(`Can't reach the backend (${e instanceof Error ? e.message : "connection error"}). Check the Render service and its CORS_ORIGINS setting.`);
    }
  }
  useEffect(() => { void loadMedia(); }, []);

  const filtered = useMemo(() => items.filter((item) => {
    const matchesType = filter === "All" || item.media_type === filter;
    const q = search.toLowerCase();
    return matchesType && (!q || `${item.filename} ${item.description || ""}`.toLowerCase().includes(q));
  }), [items, search, filter]);
  const images = items.filter((x) => x.media_type === "Image").length;
  const videos = items.filter((x) => x.media_type === "Video").length;
  const bytes = items.reduce((n, x) => n + (x.size_bytes || 0), 0);
  const mediaUrl = (item: Media) => item.url?.startsWith("http") ? item.url : `${API}${item.url || ""}`;

  async function uploadFiles(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setBusy(true); setStatus("");
    let success = 0; const errors: string[] = [];
    for (const file of files) {
      const form = new FormData(); form.append("file", file);
      try {
        const r = await fetch(`${API}/media`, { method: "POST", body: form });
        if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || `HTTP ${r.status}`); }
        success++;
      } catch (err) { errors.push(`${file.name}: ${err instanceof Error ? err.message : "upload failed"}`); }
    }
    setStatus(`${success} file(s) uploaded.${errors.length ? ` Errors: ${errors.join("; ")}` : ""}`);
    await loadMedia(); setBusy(false); e.target.value = "";
  }
  async function runAiSearch() {
    if (!aiQuery.trim()) { setStatus("Enter a description to search."); return; }
    setBusy(true); setStatus("Searching indexed media…");
    try {
      const r = await fetch(`${API}/search/ai?q=${encodeURIComponent(aiQuery.trim())}`);
      const data = await r.json(); if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setAiResults(data.items || []); setStatus(`${data.count ?? (data.items || []).length} result(s) found.`);
    } catch (e) { setStatus(`AI search failed: ${e instanceof Error ? e.message : "connection error"}`); }
    setBusy(false);
  }
  async function runVisualSearch() {
    if (!visualFile) { setStatus("Choose a reference image first."); return; }
    setBusy(true); setStatus("Analyzing reference image…");
    const form = new FormData(); form.append("file", visualFile);
    try {
      const r = await fetch(`${API}/search/visual`, { method: "POST", body: form });
      const data = await r.json(); if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      const list: Media[] = ((data.items || []) as Media[]).slice().sort((a, b) => (b.match_percent ?? 0) - (a.match_percent ?? 0));
      setAiResults(list);
      if (!list.length) setStatus(data.not_indexed ? `No comparable images found. ${data.not_indexed} saved image(s) are not indexed yet - press "Re-index saved images".` : "No matching images found. Upload some images first.");
      else setStatus(`${list.length} image(s) ranked by visual similarity (highest match first).${data.not_indexed ? ` ${data.not_indexed} saved image(s) are not indexed yet.` : ""}`);
    } catch (e) { setStatus(`Visual search failed: ${e instanceof Error ? e.message : "connection error"}`); }
    setBusy(false);
  }
  async function runReindex() {
    setBusy(true); setStatus("Re-indexing saved images…");
    try {
      const r = await fetch(`${API}/media/reindex`, { method: "POST" });
      const data = await r.json(); if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setStatus(`Re-indexed ${data.updated} image(s).${data.failed_ids?.length ? ` ${data.failed_ids.length} could not be read (missing or invalid files).` : ""}`);
    } catch (e) { setStatus(`Re-index failed: ${e instanceof Error ? e.message : "connection error"}`); }
    setBusy(false);
  }
  async function deleteItem(item: Media) {
    if (!confirm(`Remove ${item.filename}?`)) return;
    try {
      const r = await fetch(`${API}/media/${item.id}`, { method: "DELETE" });
      const data = await r.json().catch(() => ({})); if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setStatus(`${item.filename} removed.`); await loadMedia();
    } catch (e) { setStatus(`Delete failed: ${e instanceof Error ? e.message : "error"}`); }
  }

  function cards(list: Media[]) {
    if (!list.length) return <div className="empty">No matching media found. Upload some files to get started.</div>;
    return <div className="media-grid">{list.map((item) => <article className="media-card" key={item.id}>
      {item.media_type === "Image" ? <img className="preview" src={mediaUrl(item)} alt={item.filename} /> : <video className="preview" src={mediaUrl(item)} controls />}
      <div className="card-body"><strong>{item.filename}</strong><p>{item.media_type} · {((item.size_bytes || 0) / 1024).toFixed(1)} KB</p>
      {item.description && item.description !== item.filename && <details><summary>AI description</summary><p>{item.description}</p></details>}
      {item.match_percent !== undefined && <div className="match"><strong>Match: {Number(item.match_percent).toFixed(1)}%</strong><div className="match-bar"><span style={{ width: `${Math.min(100, Math.max(0, Number(item.match_percent)))}%` }} /></div></div>}
      <button className="button secondary small" onClick={() => void deleteItem(item)}>Remove</button></div>
    </article>)}</div>;
  }

  return <main className="shell">
    <header className="topbar"><div className="brand"><div className="brand-title">◈ Media<span>Vault</span></div><div className="brand-caption">MULTIMEDIA DATABASE SYSTEM</div></div>
      <nav>{pages.map((p) => <button key={p} onClick={() => { setPage(p); setStatus(""); }} className={`nav-button ${page === p ? "active" : ""}`}>{p}</button>)}</nav>
    </header>
    <div className="rule" />
    {apiOk === false && <div className="notice error">Backend connection issue: {status || "Render API is not responding."}</div>}
    {status && page !== "Home" && <div className={`notice ${status.toLowerCase().includes("failed") || status.toLowerCase().includes("can't") ? "error" : ""}`}>{status}</div>}

    {page === "Home" && <>
      <section className="home-hero"><div className="hero-copy"><div className="eyebrow">YOUR DIGITAL ASSET HUB</div><h1>Your Media.<br/><span>Organized Smarter.</span></h1><p>Upload, discover, preview and organize images and videos in a persistent multimedia workspace powered by SQLite and AI search.</p>
        <div className="actions"><button className="button primary" onClick={() => setPage("Media Library")}>Explore Media →</button><button className="button secondary" onClick={() => setPage("Upload Media")}>＋ Upload Your Files</button><button className="button secondary" onClick={() => setPage("AI Search")}>✨ Try AI Search</button></div></div>
        <div className="hero-art-card"><div className="big-icon">▣</div><h2>MediaVault</h2><div className="eyebrow">STORE · DISCOVER · ORGANIZE</div></div></section>
      <section><h2 className="section-title">Dashboard Overview</h2><p className="section-sub">A live snapshot of your saved media collection.</p>
        <div className="metrics"><Metric icon="🗂️" label="Total Assets" value={items.length} note="Files in database"/><Metric icon="🖼️" label="Images" value={images} note="Image assets"/><Metric icon="🎬" label="Videos" value={videos} note="Video assets"/><Metric icon="💾" label="Storage Used" value={`${(bytes / (1024 * 1024)).toFixed(2)} MB`} note="Uploaded file size"/></div></section>
      <section><h2 className="section-title">Your Workspace</h2><div className="feature-grid"><Feature icon="🔎" title="Explore Your Media" text="Search filenames, inspect AI descriptions and filter by media type." action="Open Library →" onClick={() => setPage("Media Library")}/><Feature icon="📤" title="Upload Center" text="Save images and videos to disk storage with SQLite metadata." action="Upload Media →" onClick={() => setPage("Upload Media")}/><Feature icon="🧠" title="AI Visual Search" text="Find media using natural language or a reference image." action="Search with AI →" onClick={() => setPage("AI Search")}/></div></section>
    </>}

    {page === "Media Library" && <><Hero eyebrow="YOUR DIGITAL COLLECTION" title={<>Explore Your <span>Media.</span></>} text="Browse your saved images and videos. Search filenames or AI-generated descriptions."/><div className="toolbar"><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search filename or image description…"/><select value={filter} onChange={(e) => setFilter(e.target.value)}><option value="All">All Files</option><option value="Image">Images</option><option value="Video">Videos</option></select><button className="button secondary" onClick={() => void loadMedia()}>Refresh</button></div><p className="section-sub">{filtered.length} matching asset(s)</p>{cards(filtered)}</>}

    {page === "Upload Media" && <><Hero eyebrow="UPLOAD CENTER" title={<>Add Your <span>Media.</span></>} text="Upload images and videos to your MediaVault library. Supported formats: PNG, JPG, JPEG, WEBP, GIF, MP4, MOV, WEBM and AVI."/><div className="panel upload-panel"><label className="upload-box"><span className="upload-icon">⇧</span><strong>{busy ? "Uploading files…" : "Choose images or videos"}</strong><span>Click to select one or more files · up to 50 MB each</span><input type="file" multiple accept="image/png,image/jpeg,image/webp,image/gif,video/mp4,video/quicktime,video/webm,video/x-msvideo" onChange={(e) => void uploadFiles(e)} disabled={busy}/></label>{status && <p className="status-text">{status}</p>}</div></>}

    {page === "AI Search" && <><Hero eyebrow="AI-POWERED DISCOVERY" title={<>Search by <span>Meaning.</span></>} text="Describe what you remember seeing, or upload a reference image. MediaVault searches indexed descriptions and ranks relevant results."/><div className="panel"><h2>🔎 Natural-language search</h2><p className="muted">Examples: a dog running on grass, students in a classroom, a red car near a mountain.</p><div className="search-row"><input value={aiQuery} onChange={(e) => setAiQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") void runAiSearch(); }} placeholder="Describe the media you want to find…"/><button className="button primary" disabled={busy} onClick={() => void runAiSearch()}>Search Media with AI</button></div><hr className="divider"/><h2>🖼️ Search by reference image</h2><p className="muted">Choose an image and see every saved image ranked by how closely it matches.</p><input type="file" accept="image/png,image/jpeg,image/webp,image/gif" onChange={(e) => setVisualFile(e.target.files?.[0] || null)}/><button className="button primary" disabled={busy} onClick={() => void runVisualSearch()}>{busy ? "Searching…" : "Find Visually Similar Media"}</button> <button className="button secondary" disabled={busy} onClick={() => void runReindex()}>Re-index saved images</button></div>{aiResults.length > 0 && <><h2 className="section-title">Search Results</h2>{cards(aiResults)}</>}</>}

    {page === "About" && <><Hero eyebrow="ABOUT THE PROJECT" title={<>Multimedia Database<br/><span>Management System</span></>} text="A multimedia management platform combining database storage with AI-assisted discovery."/><div className="panel about"><h2>Project Overview</h2><p>MediaVault stores uploaded images and videos, records their metadata in SQLite, and exposes a FastAPI backend. Optional AI features generate image descriptions and embeddings for natural-language and reference-image search.</p><h2>Main Features</h2><ul><li>Persistent SQLite metadata storage</li><li>Upload, preview, browse and delete media</li><li>Search and filter by filename, description and media type</li><li>AI-generated image descriptions</li><li>Semantic search using natural-language queries</li><li>Reference-image visual similarity search</li></ul><h2>Technology Stack</h2><p>Next.js frontend · FastAPI backend · SQLite database · file storage · local image-similarity matching with optional Gemini text search</p></div></>}
    <footer className="footer">◈ MediaVault · Multimedia Database Management System <span>Store · Discover · Organize</span></footer>
  </main>;
}
function Metric({icon,label,value,note}:{icon:string;label:string;value:string|number;note:string}) { return <div className="metric"><div className="metric-icon">{icon}</div><div className="metric-label">{label}</div><div className="metric-number">{value}</div><div className="metric-note">{note}</div></div>; }
function Feature({icon,title,text,action,onClick}:{icon:string;title:string;text:string;action:string;onClick:()=>void}) { return <div className="feature-card"><div className="feature-icon">{icon}</div><div className="feature-name">{title}</div><p>{text}</p><button className="button secondary small" onClick={onClick}>{action}</button></div>; }
function Hero({eyebrow,title,text}:{eyebrow:string;title:React.ReactNode;text:string}) { return <section className="page-hero"><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{text}</p></section>; }
