from pathlib import Path
from dotenv import load_dotenv
import os
import time
import json
import math
import re
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
from openai import AsyncOpenAI

# =========================
# Load .env (local only)
# =========================
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    load_dotenv(env_path)

# =========================
# Config
# =========================
OWNER_NOTIFY_EMAIL = os.getenv("OWNER_NOTIFY_EMAIL", "").strip()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", "").strip()  # optional; if empty we'll use OWNER_NOTIFY_EMAIL

SERPAPI_KEY = os.getenv("SEARCHAPI_KEY", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

SITE_SITEMAP_URL = os.getenv("SITE_SITEMAP_URL", "").strip()
SITE_BASE_URL = os.getenv("SITE_BASE_URL", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
CONTACT_FORM_URL = os.getenv("CONTACT_FORM_URL", "").strip()
CONTACT_FORM_URL_EN = os.getenv("CONTACT_FORM_URL_EN", "https://www.aboodfreediver.com/form1.php").strip()
CONTACT_FORM_URL_DE = os.getenv("CONTACT_FORM_URL_DE", "https://www.aboodfreediver.com/form1de.php").strip()
CONTACT_FORM_URL_AR = os.getenv("CONTACT_FORM_URL_AR", "https://www.aboodfreediver.com/form1ar.php").strip()
CONTACT_FORM_PATH = os.getenv("CONTACT_FORM_PATH", "/form1-contact").strip()

DB_PATH = os.getenv("DB_PATH", "data.sqlite3")

MAX_URLS = int(os.getenv("MAX_URLS", "150"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "800"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.20"))

# CORS
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    # Safer than "*" if you use cookies/credentials. Add your real domains here.
    ALLOWED_ORIGINS = ["http://localhost:5500", "http://localhost:3000"]

# Approval token
ADMIN_TOKEN_SECRET = os.getenv("ADMIN_TOKEN_SECRET", "").strip() or ("CHANGE_ME_" + secrets.token_urlsafe(16))
serializer = URLSafeSerializer(ADMIN_TOKEN_SECRET, salt="booking-approval")

# OpenAI client
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            title TEXT,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS human_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            text TEXT NOT NULL,
            ts INTEGER NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS booking_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            status TEXT NOT NULL, -- pending|approved|denied
            details_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
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
    ql = (q or "").lower()
    triggers = [
        "book",
        "booking",
        "reserve",
        "reservation",
        "schedule",
        "join course",
        "lesson",
        "class",
        "training",
        "session",
        "availability",
        "price",
        "tomorrow",
    ]
    return any(t in ql for t in triggers)


def wants_human(q: str) -> bool:
    ql = (q or "").lower()
    triggers = ["human", "real person", "abood", "instructor", "call me", "contact", "whatsapp", "agent"]
    return any(t in ql for t in triggers)


def is_medical_or_high_risk(q: str) -> bool:
    ql = (q or "").lower()
    triggers = ["faint", "blackout", "lung", "pain", "injury", "blood", "doctor", "medical", "pregnan", "asthma"]
    return any(t in ql for t in triggers)


# =========================
# OpenAI helpers
# =========================
async def embed_text(text: str) -> List[float]:
    try:
        resp = await openai_client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=(text or "")[:6000],
        )
        return list(resp.data[0].embedding)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI embeddings failed: {str(e)}")


async def generate_answer(system: str, user: str) -> str:
    # Use chat.completions for gpt-4o-mini (stable + supports temperature)
    try:
        resp = await openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI chat failed: {str(e)}")


# =========================
# SendGrid email
# =========================
def send_email(to_email: str, subject: str, html: str):
    if not SENDGRID_API_KEY:
        raise RuntimeError("SENDGRID_API_KEY not set")
    if not to_email:
        raise RuntimeError("OWNER_NOTIFY_EMAIL not set")

    from_email = SMTP_FROM or to_email
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }
    headers = {"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"}
    r = requests.post("https://api.sendgrid.com/v3/mail/send", headers=headers, data=json.dumps(payload), timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"SendGrid error {r.status_code}: {r.text}")


# =========================
# SerpAPI (safe)
# =========================
def serpapi_search(query: str) -> List[Dict[str, str]]:
    if not SERPAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": 5},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        out = []
        for item in data.get("organic_results", [])[:5]:
            out.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return out
    except Exception:
        return []


# =========================
# Indexing (sitemap -> pages -> chunks -> embeddings)
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
    if not SITE_SITEMAP_URL:
        print("SITE_SITEMAP_URL not set; skipping indexing.")
        return

    print("Fetching sitemap:", SITE_SITEMAP_URL)
    r = requests.get(SITE_SITEMAP_URL, timeout=30)
    r.raise_for_status()

    urls = parse_sitemap_urls(r.text, SITE_BASE_URL)
    print(f"Sitemap URLs: {len(urls)} (capped by MAX_URLS)")

    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    conn.commit()

    for idx, url in enumerate(urls):
        try:
            pr = requests.get(url, timeout=25, headers={"User-Agent": "AboodFreediverBot/1.0"})
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
                    (url, title, ch, json.dumps(emb)),
                )
            conn.commit()
        except Exception as e:
            print("Index error:", url, str(e))

        if (idx + 1) % 10 == 0:
            print(f"Indexed {idx+1}/{len(urls)} pages")

    conn.close()
    print("Index build complete.")


async def retrieve_site_context(question: str) -> Tuple[List[Dict[str, Any]], float]:
    if chunks_count() == 0:
        return [], 0.0

    q_emb = await embed_text(question)

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT url,title,text,embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()

    scored = []
    for r in rows:
        try:
            emb = json.loads(r["embedding"])
            score = cosine(q_emb, emb)
            scored.append((score, r["url"], r["title"], r["text"]))
        except Exception:
            continue

    if not scored:
        return [], 0.0

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_K]
    confidence = float(top[0][0]) if top else 0.0

    chunks = [{"score": s, "url": u, "title": (t or ""), "text": tx} for (s, u, t, tx) in top]
    return chunks, confidence


# =========================
# Sessions + human messages
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
    return [{"role": "Abood Freediver Team", "text": r["text"], "ts": r["ts"]} for r in rows]


# =========================
# Booking flow
# =========================
def create_booking_request(session_id: str, details: Dict[str, Any]) -> int:
    conn = db()
    cur = conn.cursor()
    ts = now_ts()
    cur.execute(
        "INSERT INTO booking_requests(session_id,status,details_json,created_at,updated_at) VALUES(?,?,?,?,?)",
        (session_id, "pending", json.dumps(details), ts, ts),
    )
    conn.commit()
    booking_id = cur.lastrowid
    conn.close()
    return int(booking_id)


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

def get_latest_booking_for_session(session_id: str) -> Optional[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM booking_requests WHERE session_id=? ORDER BY created_at DESC, id DESC LIMIT 1",
        (session_id,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def detect_language_from_text(text: str) -> str:
    """Return 'ar', 'de', or 'en' based on simple heuristics."""
    t = (text or "").strip()
    if not t:
        return "en"

    # Arabic unicode ranges
    if re.search(r"[\u0590-\u05FF\u0600-\u06FF\u0700-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", t):
        return "ar"

    # German hints (umlauts / ß / common words)
    if re.search(r"[äöüÄÖÜß]", t):
        return "de"
    lowered = t.lower()
    german_words = [
        "guten", "hallo", "bitte", "danke", "ich", "du", "sie", "wir", "und",
        "möchte", "moechte", "kann", "können", "koennen", "zeit", "datum",
        "uhr", "preis", "kurs", "training"
    ]
    if any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in german_words):
        return "de"

    return "en"


def detect_language(req_question: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Detect language from current question + recent user history."""
    # Prefer current question
    lang = detect_language_from_text(req_question)
    if lang != "en":
        return lang

    # Check last few user messages
    for m in reversed(history or []):
        if (m or {}).get("role") in ("user", "human"):
            lang2 = detect_language_from_text((m or {}).get("content") or (m or {}).get("text") or "")
            if lang2 != "en":
                return lang2

    return "en"


def build_contact_form_url(page_url: str = "", lang: str = "en") -> str:
    """Return the correct contact form URL.

    Priority:
      1) CONTACT_FORM_URL (single override)
      2) Language-specific URLs (CONTACT_FORM_URL_EN/DE/AR)
      3) SITE_BASE_URL + CONTACT_FORM_PATH (fallback)
      4) Derive origin from page_url + CONTACT_FORM_PATH
      5) CONTACT_FORM_PATH
    """
    # 1) Single override
    if CONTACT_FORM_URL:
        return CONTACT_FORM_URL

    # 2) Language-specific
    if (lang or "").lower().startswith("ar"):
        return CONTACT_FORM_URL_AR
    if (lang or "").lower().startswith("de"):
        return CONTACT_FORM_URL_DE

    # default English
    if CONTACT_FORM_URL_EN:
        return CONTACT_FORM_URL_EN

    # 3/4/5) Backward compatible fallback
    if SITE_BASE_URL:
        return SITE_BASE_URL.rstrip("/") + "/" + CONTACT_FORM_PATH.lstrip("/")

    # Fallback: try deriving from page_url origin
    if page_url:
        m = re.match(r"^(https?://[^/]+)", page_url.strip())
        if m:
            return m.group(1).rstrip("/") + "/" + CONTACT_FORM_PATH.lstrip("/")
    return CONTACT_FORM_PATH


def make_approval_links(base_url: str, booking_id: int) -> Tuple[str, str]:
    token = serializer.dumps({"booking_id": booking_id})
    approve = f"{base_url}/admin/booking/approve?token={token}"
    deny = f"{base_url}/admin/booking/deny?token={token}"
    return approve, deny


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Abood Freediver Chat Backend")

# If you ever set "*" origins, credentials must be False.
allow_credentials = True
if "*" in ALLOWED_ORIGINS:
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=allow_credentials,
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
    booking_next_url: Optional[str] = None
    sources: Optional[List[Dict[str, str]]] = None  # you can ignore on frontend


index_task = None


@app.on_event("startup")
async def on_startup():
    global index_task
    init_db()
    # Build index in background only if empty and sitemap configured
    if chunks_count() == 0 and SITE_SITEMAP_URL:
        index_task = asyncio.create_task(build_index())


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question or not req.session_id:
        raise HTTPException(status_code=400, detail="question and session_id required")

    ensure_session(req.session_id)
    q = req.question.strip()

    # Human takeover / safety
    if is_medical_or_high_risk(q) or wants_human(q):
        if OWNER_NOTIFY_EMAIL and SENDGRID_API_KEY:
            try:
                send_email(
                    OWNER_NOTIFY_EMAIL,
                    "Abood Freediver: Human help requested",
                    f"""
                    <h3>Human help requested</h3>
                    <p><b>Session:</b> {req.session_id}</p>
                    <p><b>Page:</b> {req.page_url or ''}</p>
                    <p><b>User message:</b> {q}</p>
                    """,
                )
            except Exception as e:
                print("Email notify failed:", str(e))

        return ChatResponse(
            answer=(
                "Thanks — this is the Abood Freediver team. "
                "For safety and accuracy, Abood will reply directly. "
                "Please share your name and preferred contact (WhatsApp or email)."
            ),
            needs_human=True,
            booking_pending=False,
            sources=[],
        )

    
    # If there is an existing booking for this session, reuse its status instead of creating duplicates.
    latest_booking = get_latest_booking_for_session(req.session_id)
    if latest_booking:
        status = (latest_booking.get("status") or "").lower()
        details = {}
        try:
            details = json.loads(latest_booking.get("details_json") or "{}")
        except Exception:
            details = {}

        if status == "pending" and looks_like_booking(q):
            return ChatResponse(
                answer=(
                    "Your booking request is still pending Abood’s approval. "
                    "If you want to update anything (date/time, number of people, contact), "
                    "tell us here and we will forward it."
                ),
                needs_human=True,
                booking_pending=True,
                booking_next_url=None,
                sources=[],
            )

        if status == "approved":
            lang = detect_language(q, req.history)
            contact_url = build_contact_form_url(details.get("page_url", req.page_url or ""), lang=lang)
            if looks_like_booking(q):
                return ChatResponse(
                    answer=(
                        "Approved — please continue the booking on our contact form here: "
                        f"{contact_url}\n\n"
                        "After you submit the form, you can keep chatting here if you have questions."
                    ),
                    needs_human=False,
                    booking_pending=False,
                    booking_next_url=contact_url,
                    sources=[],
                )

        if status == "denied":
            if looks_like_booking(q):
                return ChatResponse(
                    answer=(
                        "Sorry — we couldn’t approve that booking request. "
                        "If you share alternative dates/times (and your experience level), we can check again. "
                        "You can also ask any other questions here."
                    ),
                    needs_human=False,
                    booking_pending=False,
                    booking_next_url=None,
                    sources=[],
                )

# Booking request -> email notify + pending
    if looks_like_booking(q):
        details = {"question": q, "page_url": req.page_url or "", "history": req.history or []}
        booking_id = create_booking_request(req.session_id, details)

        if OWNER_NOTIFY_EMAIL and SENDGRID_API_KEY:
            approve, deny = ("", "")
            if PUBLIC_BASE_URL:
                approve, deny = make_approval_links(PUBLIC_BASE_URL, booking_id)
            try:
                send_email(
                    OWNER_NOTIFY_EMAIL,
                    "Abood Freediver: Booking request pending approval",
                    f"""
                    <h3>Booking request pending approval</h3>
                    <p><b>Session:</b> {req.session_id}</p>
                    <p><b>Page:</b> {req.page_url or ''}</p>
                    <p><b>User message:</b> {q}</p>
                    <p><b>Approve:</b> <a href="{approve}">{approve}</a></p>
                    <p><b>Deny:</b> <a href="{deny}">{deny}</a></p>
                    """,
                )
            except Exception as e:
                print("Email booking notify failed:", str(e))

        return ChatResponse(
            answer=(
                "This is the Abood Freediver team. I can take your booking request, but I can’t confirm it until Abood approves.\n\n"
                "Please share:\n"
                "1) Desired date(s) + time window\n"
                "2) Course/session type\n"
                "3) Number of people\n"
                "4) Your contact (WhatsApp or email)\n\n"
                "I’ve sent your request to Abood for approval."
            ),
            needs_human=True,
            booking_pending=True,
            booking_next_url=None,
            sources=[],
        )

    # Retrieval
    chunks, conf = await retrieve_site_context(q)

    # Make the assistant always speak as Abood Freediver team
    system = (
        "You are Aqua, the official assistant of Abood Freediver (freediving in Aqaba).\n"
        "Identity rules:\n"
        "- Speak as a member of the Abood Freediver team (use 'we' when appropriate).\n"
        "- Do not claim you personally are Abood; you are the team's assistant.\n"
        "Behavior rules:\n"
        "- Prefer answering using SITE_CONTEXT.\n"
        "- If SITE_CONTEXT is insufficient, you may use WEB_CONTEXT if provided.\n"
        "- Never confirm a booking; only say it’s pending Abood’s approval.\n"
        "- Keep answers concise, practical, and safety-conscious.\n"
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
        "- Answer as the Abood Freediver team assistant.\n"
        "- If info is missing, ask 1–2 quick questions or suggest contacting us.\n"
    )

    answer = await generate_answer(system, prompt)

    # Keep returning sources for debugging; your frontend can simply not display them.
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

    details = {}
    try:
        details = json.loads(booking.get("details_json") or "{}")
    except Exception:
        details = {}

    lang = detect_language(details.get("question", ""), details.get("history") or [])

    contact_url = build_contact_form_url(details.get("page_url", ""), lang=lang)

    # This message is shown to the user via /chat/status polling
    add_human_message(
        booking["session_id"],
        f"Approved ✅\nPlease continue the booking on our contact form: {contact_url}\nAfter you submit the form, you can keep chatting here if you have any questions.",
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
        "Sorry — we can’t approve this booking request right now.\nIf you share alternative dates/times (and your experience level), we can check again.\nIf you have any other questions, you can ask here.",
    )
    return {"ok": True, "status": "denied", "booking_id": booking_id}



@app.get("/admin/index/status")
def index_status():
    return {"chunks": chunks_count()}


@app.post("/admin/reindex")
async def admin_reindex():
    global index_task

    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()

    if index_task and not index_task.done():
        return {"ok": True, "status": "already_running"}

    index_task = asyncio.create_task(build_index())
    return {"ok": True, "status": "started"}


@app.get("/admin/reindex/status")
def admin_reindex_status():
    running = bool(index_task and not index_task.done())
    return {"running": running, "chunks": chunks_count()}
