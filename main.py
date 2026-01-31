from pathlib import Path
from dotenv import load_dotenv

# Always load .env located next to this file (reliable on Windows/Render)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

import os
import time
import json
import math
import sqlite3
import secrets
import asyncio
from typing import List, Dict, Any, Optional, Tuple

import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bs4 import BeautifulSoup
from itsdangerous import URLSafeSerializer, BadSignature


# =========================
# Config (static / safe)
# =========================
OWNER_NOTIFY_EMAIL = os.getenv("OWNER_NOTIFY_EMAIL", "").strip()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "").strip()

# Web search (SerpAPI)
SERPAPI_KEY = os.getenv("SEARCHAPI_KEY", "").strip()

# LLM keys (choose one)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# CORS
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:5500", "http://localhost:3000"]

# Security for approval links
ADMIN_TOKEN_SECRET = os.getenv("ADMIN_TOKEN_SECRET", "").strip()
if not ADMIN_TOKEN_SECRET:
    ADMIN_TOKEN_SECRET = "CHANGE_ME_" + secrets.token_urlsafe(16)

serializer = URLSafeSerializer(ADMIN_TOKEN_SECRET, salt="booking-approval")

DB_PATH = os.getenv("DB_PATH", "data.sqlite3")

MAX_URLS = int(os.getenv("MAX_URLS", "150"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "800"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.20"))


# =========================
# Database
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        title TEXT,
        text TEXT NOT NULL,
        embedding TEXT NOT NULL
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS human_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        text TEXT NOT NULL,
        ts INTEGER NOT NULL
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS booking_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        status TEXT NOT NULL, -- pending|approved|denied
        details_json TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      )
    """)
    conn.commit()
    conn.close()

def chunks_count() -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM chunks")
    n = cur.fetchone()["n"]
    conn.close()
    return int(n)


# =========================
# Utilities
# =========================
def now_ts() -> int:
    return int(time.time())

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return (dot / denom) if denom else 0.0

def strip_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    cleaned = "\n".join(lines)
    return title, cleaned

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < MAX_CHUNKS:
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
        if end == len(text):
            break
    return chunks

def looks_like_booking(q: str) -> bool:
    ql = q.lower()
    triggers = [
        "book", "booking", "reserve", "reservation", "schedule",
        "join course", "course booking", "lesson", "class",
        "training", "session", "availability", "price for course"
    ]
    return any(t in ql for t in triggers)

def wants_human(q: str) -> bool:
    ql = q.lower()
    triggers = ["human", "real person", "instructor", "call me", "contact you", "talk to you", "agent"]
    return any(t in ql for t in triggers)

def is_medical_or_high_risk(q: str) -> bool:
    ql = q.lower()
    triggers = ["faint", "blackout", "lung", "pain", "injury", "blood", "doctor", "medical", "pregnan", "asthma"]
    return any(t in ql for t in triggers)


# =========================
# LLM (OpenAI or Gemini)
# =========================
async def embed_text(text: str) -> List[float]:
    if OPENAI_API_KEY:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            input=text[:6000]
        )
        return list(resp.data[0].embedding)

    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        # MUST start with "models/"
        model = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

        res = genai.embed_content(
    model=model,
    content=text[:6000],
    task_type="retrieval_query"
)

# Support both dict and object return shapes
        if isinstance(res, dict):
         emb = res.get("embedding")
        else:
         emb = getattr(res, "embedding", None)

        if not emb:
         raise HTTPException(status_code=500, detail="Gemini embedding response missing 'embedding'")

        return list(emb)


    raise HTTPException(status_code=500, detail="No embedding key set. Set OPENAI_API_KEY or GEMINI_API_KEY.")

async def generate_answer(system: str, user: str) -> str:
    if OPENAI_API_KEY:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()

    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash"))
        resp = model.generate_content(
            [system, user],
            generation_config={"temperature": 0.2}
        )
        return resp.text.strip()

    raise HTTPException(status_code=500, detail="No chat key set. Set OPENAI_API_KEY or GEMINI_API_KEY.")


# =========================
# Web search (SerpAPI)
# =========================
def serpapi_search(query: str) -> List[Dict[str, str]]:
    if not SERPAPI_KEY:
        return []
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": 5}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("organic_results", [])[:5]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", "")
        })
    return results


# =========================
# Email (SendGrid)
# =========================
def send_email(to_email: str, subject: str, html: str):
    if not SENDGRID_API_KEY:
        raise RuntimeError("SENDGRID_API_KEY not set")
    if not to_email:
        raise RuntimeError("OWNER_NOTIFY_EMAIL not set")

    url = "https://api.sendgrid.com/v3/mail/send"
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": os.getenv("SMTP_FROM", to_email) or to_email},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }
    headers = {"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"SendGrid error {r.status_code}: {r.text}")


# =========================
# Site indexing (sitemap -> pages -> chunks -> embeddings)
# =========================
def parse_sitemap_urls(xml_text: str, site_base_url: str) -> List[str]:
    soup = BeautifulSoup(xml_text, "xml")
    locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
    seen = set()
    urls = []
    for u in locs:
        if u in seen:
            continue
        if site_base_url and not u.startswith(site_base_url):
            continue
        seen.add(u)
        urls.append(u)
        if len(urls) >= MAX_URLS:
            break
    return urls

async def build_index():
    site_sitemap_url = os.getenv("SITE_SITEMAP_URL", "").strip()
    site_base_url = os.getenv("SITE_BASE_URL", "").strip()

    if not site_sitemap_url:
        print("SITE_SITEMAP_URL not set; skipping indexing.")
        return

    print("Fetching sitemap:", site_sitemap_url)
    r = requests.get(site_sitemap_url, timeout=30)
    r.raise_for_status()

    urls = parse_sitemap_urls(r.text, site_base_url)
    print(f"Sitemap URLs: {len(urls)} (capped by MAX_URLS)")

    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    conn.commit()

    for idx, url in enumerate(urls):
        try:
            pr = requests.get(url, timeout=25, headers={"User-Agent": "FreediveChatBot/1.0"})
            if pr.status_code >= 400:
                continue
            title, text = strip_text(pr.text)
            if not text or len(text) < 200:
                continue

            chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
            for ch in chunks:
                emb = await embed_text(ch)
                cur.execute(
                    "INSERT INTO chunks(url,title,text,embedding) VALUES(?,?,?,?)",
                    (url, title, ch, json.dumps(emb))
                )
            conn.commit()
        except Exception as e:
            print("Index error:", url, str(e))

        if idx % 10 == 0:
            print(f"Indexed {idx+1}/{len(urls)} pages")

    conn.close()
    print("Index build complete.")

async def retrieve_site_context(question: str) -> Tuple[List[Dict[str, Any]], float]:
    q_emb = await embed_text(question)

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT url,title,text,embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()

    scored = []
    for r in rows:
        emb = json.loads(r["embedding"])
        score = cosine(q_emb, emb)
        scored.append((score, r["url"], r["title"], r["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_K]
    confidence = top[0][0] if top else 0.0

    chunks = []
    for score, url, title, text in top:
        chunks.append({"score": score, "url": url, "title": title or "", "text": text})
    return chunks, confidence


# =========================
# Session + human messages
# =========================
def ensure_session(session_id: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions(session_id, created_at) VALUES(?,?)", (session_id, now_ts()))
    conn.commit()
    conn.close()

def add_human_message(session_id: str, text: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT INTO human_messages(session_id, text, ts) VALUES(?,?,?)", (session_id, text, now_ts()))
    conn.commit()
    conn.close()

def get_human_messages(session_id: str) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT text, ts FROM human_messages WHERE session_id=? ORDER BY ts ASC", (session_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": "human", "text": r["text"], "ts": r["ts"]} for r in rows]


# =========================
# Booking flow
# =========================
def create_booking_request(session_id: str, details: Dict[str, Any]) -> int:
    conn = db()
    cur = conn.cursor()
    ts = now_ts()
    cur.execute(
        "INSERT INTO booking_requests(session_id,status,details_json,created_at,updated_at) VALUES(?,?,?,?,?)",
        (session_id, "pending", json.dumps(details), ts, ts)
    )
    conn.commit()
    booking_id = cur.lastrowid
    conn.close()
    return booking_id

def update_booking_status(booking_id: int, status: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE booking_requests SET status=?, updated_at=? WHERE id=?", (status, now_ts(), booking_id))
    conn.commit()
    conn.close()

def get_booking(booking_id: int) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM booking_requests WHERE id=?", (booking_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    return dict(row)

def make_approval_links(base_url: str, booking_id: int) -> Tuple[str, str]:
    token = serializer.dumps({"booking_id": booking_id})
    approve = f"{base_url}/admin/booking/approve?token={token}"
    deny = f"{base_url}/admin/booking/deny?token={token}"
    return approve, deny


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Freediving Site Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: str
    history: Optional[List[Dict[str, Any]]] = None
    page_url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    needs_human: bool = False
    booking_pending: bool = False
    sources: Optional[List[Dict[str, str]]] = None


index_task = None  # global


@app.on_event("startup")
async def on_startup():
    global index_task
    init_db()

    # Start immediately, index only if empty
    if chunks_count() == 0:
        index_task = asyncio.create_task(build_index())


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question or not req.session_id:
        raise HTTPException(status_code=400, detail="question and session_id required")
    ensure_session(req.session_id)

    q = req.question.strip()

    if is_medical_or_high_risk(q) or wants_human(q):
        if OWNER_NOTIFY_EMAIL and SENDGRID_API_KEY:
            try:
                send_email(
                    OWNER_NOTIFY_EMAIL,
                    "Human takeover requested (Freediving chat)",
                    f"<p>Session: <b>{req.session_id}</b></p><p>Page: {req.page_url or ''}</p><p>Question: {q}</p>"
                )
            except Exception:
                pass

        return ChatResponse(
            answer="I’ve notified the instructor for direct help. Please share your name and contact details if you want a reply by email or WhatsApp.",
            needs_human=True,
            booking_pending=False,
            sources=[]
        )

    if looks_like_booking(q):
        details = {"question": q, "page_url": req.page_url or "", "history": req.history or []}
        booking_id = create_booking_request(req.session_id, details)

        if OWNER_NOTIFY_EMAIL and SENDGRID_API_KEY:
            base = os.getenv("PUBLIC_BASE_URL", "").strip()
            approve, deny = ("", "") if not base else make_approval_links(base, booking_id)

            html = f"""
            <h3>Booking request pending approval</h3>
            <p><b>Session:</b> {req.session_id}</p>
            <p><b>Page:</b> {req.page_url or ''}</p>
            <p><b>User message:</b> {q}</p>
            <p><b>Approve:</b> <a href="{approve}">{approve}</a></p>
            <p><b>Deny:</b> <a href="{deny}">{deny}</a></p>
            """
            try:
                send_email(OWNER_NOTIFY_EMAIL, "Booking request pending approval", html)
            except Exception:
                pass

        return ChatResponse(
            answer=(
                "I can help you request a booking, but I cannot confirm it until the instructor approves.\n\n"
                "Please share:\n"
                "1) Desired date(s) and time window\n"
                "2) Course/session type (intro, coaching, fun dive, etc.)\n"
                "3) Number of people\n"
                "4) Your contact (email or WhatsApp)\n\n"
                "I’ve notified the instructor and your request is pending approval."
            ),
            needs_human=True,
            booking_pending=True,
            sources=[]
        )

    chunks, conf = await retrieve_site_context(q)

    system = (
        "You are Aqua, a freediving assistant for a freediving website.\n"
        "Rules:\n"
        "- Prefer answering using SITE_CONTEXT.\n"
        "- If SITE_CONTEXT is insufficient, you may use WEB_CONTEXT if provided.\n"
        "- Never confirm a booking; only say pending instructor approval.\n"
        "- Keep answers concise and safety-conscious.\n"
    )

    sources = []
    site_context = ""
    for c in chunks:
        sources.append({"title": c["title"] or "Website page", "url": c["url"]})
        site_context += f"\n\nSOURCE: {c['url']}\n{c['text'][:2000]}"

    web_results = serpapi_search(q) if conf < MIN_CONFIDENCE else []
    web_context = ""
    for r in web_results:
        web_context += f"\n\nWEB: {r.get('title','')}\n{r.get('link','')}\n{r.get('snippet','')}"

    prompt = (
        f"USER_QUESTION:\n{q}\n\n"
        f"SITE_CONTEXT:\n{site_context if site_context else '(none)'}\n\n"
        f"WEB_CONTEXT:\n{web_context if web_context else '(none)'}\n\n"
        "TASK:\n"
        "- Answer the user question.\n"
        "- If you used website info, cite it as 'From: <url>'.\n"
        "- If you used web info, cite as 'Source: <url>'.\n"
        "- If neither is sufficient, say what is missing and offer escalation.\n"
    )

    answer = await generate_answer(system, prompt)

    out_sources = sources[:4] if chunks and conf >= MIN_CONFIDENCE else []
    for r in web_results[:3]:
        if r.get("link"):
            out_sources.append({"title": r.get("title", "Web source"), "url": r["link"]})

    return ChatResponse(answer=answer, needs_human=False, booking_pending=False, sources=out_sources)


@app.get("/chat/status")
def chat_status(session_id: str = Query(...)):
    return {"messages": get_human_messages(session_id)}


@app.get("/admin/booking/approve")
def approve_booking(token: str = Query(...)):
    try:
        data = serializer.loads(token)
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid token")

    booking_id = int(data["booking_id"])
    booking = get_booking(booking_id)
    update_booking_status(booking_id, "approved")

    add_human_message(
        booking["session_id"],
        "Instructor: Your booking is confirmed. Please reply with your full name and preferred contact number to finalize details."
    )
    return {"ok": True, "status": "approved", "booking_id": booking_id}


@app.get("/admin/booking/deny")
def deny_booking(token: str = Query(...)):
    try:
        data = serializer.loads(token)
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid token")

    booking_id = int(data["booking_id"])
    booking = get_booking(booking_id)
    update_booking_status(booking_id, "denied")

    add_human_message(
        booking["session_id"],
        "Instructor: I can’t confirm that booking request yet. Please share alternative dates/times and your experience level."
    )
    return {"ok": True, "status": "denied", "booking_id": booking_id}


@app.get("/admin/index/status")
def index_status():
    return {"chunks": chunks_count()}


@app.post("/admin/reindex")
async def admin_reindex():
    global index_task

    if index_task and not index_task.done():
        return {"ok": True, "status": "already_running"}

    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()

    index_task = asyncio.create_task(build_index())
    return {"ok": True, "status": "started"}


@app.get("/admin/reindex/status")
def admin_reindex_status():
    running = bool(index_task and not index_task.done())
    return {"running": running, "chunks": chunks_count()}
