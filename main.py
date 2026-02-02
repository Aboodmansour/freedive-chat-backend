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
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
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
    ALLOWED_ORIGINS = ["http://localhost:5500", "http://localhost:3000"]

# Admin Basic Auth (Render)
ADMIN_USER = os.getenv("ADMIN_USER", "").strip()
ADMIN_PASS = os.getenv("ADMIN_PASS", "").strip()
security = HTTPBasic()

def require_admin(creds: HTTPBasicCredentials):
    if not (ADMIN_USER and ADMIN_PASS):
        raise HTTPException(status_code=500, detail="ADMIN_USER/ADMIN_PASS not set")
    ok = secrets.compare_digest(creds.username, ADMIN_USER) and secrets.compare_digest(creds.password, ADMIN_PASS)
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

# Approval token for email links
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
    # Human-help tickets (admin dashboard)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS support_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            status TEXT NOT NULL, -- open|closed
            user_message TEXT NOT NULL,
            page_url TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    # Session takeover state
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS session_state (
            session_id TEXT PRIMARY KEY,
            human_mode INTEGER NOT NULL, -- 0|1
            updated_at INTEGER NOT NULL
        )
        """
    )
    # Conversation log (FULL chat history per session)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,     -- user|assistant|human
            text TEXT NOT NULL,
            ts INTEGER NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_session_ts ON conversation_log(session_id, ts, id)")
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


def detect_language_from_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "en"
    if re.search(r"[\u0590-\u05FF\u0600-\u06FF\u0700-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", t):
        return "ar"
    if re.search(r"[äöüÄÖÜß]", t):
        return "de"
    lowered = t.lower()
    german_words = [
        "guten", "hallo", "bitte", "danke", "ich", "du", "sie", "wir", "und",
        "möchte", "moechte", "kann", "können", "koennen", "zeit", "datum",
        "uhr", "kurs", "training"
    ]
    if any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in german_words):
        return "de"
    return "en"


def detect_language(req_question: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
    lang = detect_language_from_text(req_question)
    if lang != "en":
        return lang
    for m in reversed(history or []):
        if (m or {}).get("role") in ("user", "human"):
            lang2 = detect_language_from_text((m or {}).get("content") or (m or {}).get("text") or "")
            if lang2 != "en":
                return lang2
    return "en"


# =========================
# Booking triggers (MULTILINGUAL)
# =========================
_BOOKING_PATTERNS = [
    # EN
    r"\bbook\b", r"\bbooking\b", r"\breserve\b", r"\breservation\b",
    # DE
    r"\bbuchen\b", r"\bbuchung\b", r"\breservieren\b", r"\breservierung\b",
    # AR
    r"حجز", r"احجز", r"أحجز", r"حجوزات", r"حجز\s*موعد",
]

_NEW_BOOKING_PATTERNS = [
    # EN
    r"\bbook again\b", r"\bbooking again\b", r"\bnew booking\b",
    r"\banother booking\b", r"\banother reservation\b",
    r"\bmake another booking\b", r"\bmake a new booking\b",
    r"\bnew reservation\b", r"\breserve again\b", r"\breservation again\b",
    # DE
    r"\bnochmal buchen\b", r"\bnoch einmal buchen\b", r"\bneue buchung\b",
    r"\bweitere buchung\b", r"\bnoch eine buchung\b", r"\bneue reservierung\b",
    r"\bweitere reservierung\b", r"\bnoch eine reservierung\b",
    r"\bnochmal reservieren\b", r"\bnoch einmal reservieren\b",
    # AR
    r"حجز جديد", r"حجز آخر", r"احجز مرة أخرى", r"أحجز مرة أخرى", r"حجز مرة أخرى", r"حجز ثاني",
]

def looks_like_booking(q: str) -> bool:
    text = q or ""
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _BOOKING_PATTERNS)

def wants_new_booking(q: str) -> bool:
    text = q or ""
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _NEW_BOOKING_PATTERNS)

def wants_human(q: str) -> bool:
    ql = (q or "").lower()
    triggers = ["human", "real person", "abood", "instructor", "call me", "contact", "whatsapp", "agent"]
    return any(t in ql for t in triggers)

def is_medical_or_high_risk(q: str) -> bool:
    ql = (q or "").lower()
    triggers = ["faint", "blackout", "lung", "pain", "injury", "blood", "doctor", "medical", "pregnan", "asthma"]
    return any(t in ql for t in triggers)

def can_send_booking_email() -> bool:
    return bool(OWNER_NOTIFY_EMAIL and SENDGRID_API_KEY)


# =========================
# Conversation log helpers
# =========================
def log_message(session_id: str, role: str, text: str):
    if not session_id or not role or text is None:
        return
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversation_log(session_id, role, text, ts) VALUES(?,?,?,?)",
        (session_id, role, text, now_ts()),
    )
    conn.commit()
    conn.close()

def get_conversation(session_id: str, limit: int = 300) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, text, ts FROM conversation_log WHERE session_id=? ORDER BY ts ASC, id ASC LIMIT ?",
        (session_id, int(limit)),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =========================
# Human takeover state + tickets
# =========================
def set_human_mode(session_id: str, human_mode: bool):
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO session_state(session_id, human_mode, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(session_id) DO UPDATE SET human_mode=excluded.human_mode, updated_at=excluded.updated_at",
        (session_id, 1 if human_mode else 0, now_ts()),
    )
    conn.commit()
    conn.close()

def is_human_mode(session_id: str) -> bool:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT human_mode FROM session_state WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    conn.close()
    return bool(row and int(row["human_mode"]) == 1)

def upsert_support_request(session_id: str, user_message: str, page_url: str):
    conn = db()
    cur = conn.cursor()
    ts = now_ts()
    cur.execute(
        "SELECT id FROM support_requests WHERE session_id=? AND status='open' ORDER BY updated_at DESC LIMIT 1",
        (session_id,),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE support_requests SET user_message=?, page_url=?, updated_at=? WHERE id=?",
            (user_message, page_url or "", ts, int(row["id"])),
        )
    else:
        cur.execute(
            "INSERT INTO support_requests(session_id,status,user_message,page_url,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (session_id, "open", user_message, page_url or "", ts, ts),
        )
    conn.commit()
    conn.close()

def close_support_request(session_id: str):
    conn = db()
    cur = conn.cursor()
    ts = now_ts()
    cur.execute(
        "UPDATE support_requests SET status='closed', updated_at=? WHERE session_id=? AND status='open'",
        (ts, session_id),
    )
    conn.commit()
    conn.close()

def list_support_requests(status: str = "open") -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM support_requests WHERE status=? ORDER BY updated_at DESC LIMIT 200",
        (status,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def list_booking_requests(status: str = "pending") -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM booking_requests WHERE status=? ORDER BY updated_at DESC LIMIT 200",
        (status,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


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
                {"title": item.get("title", ""), "link": item.get("link", ""), "snippet": item.get("snippet", "")}
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
# Booking flow helpers
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

def build_contact_form_url(page_url: str = "", lang: str = "en") -> str:
    if CONTACT_FORM_URL:
        return CONTACT_FORM_URL
    if (lang or "").lower().startswith("ar"):
        return CONTACT_FORM_URL_AR
    if (lang or "").lower().startswith("de"):
        return CONTACT_FORM_URL_DE
    if CONTACT_FORM_URL_EN:
        return CONTACT_FORM_URL_EN
    if SITE_BASE_URL:
        return SITE_BASE_URL.rstrip("/") + "/" + CONTACT_FORM_PATH.lstrip("/")
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
    sources: Optional[List[Dict[str, str]]] = None

class AdminMessageRequest(BaseModel):
    session_id: str
    text: str
    close: bool = False

index_task = None


@app.on_event("startup")
async def on_startup():
    global index_task
    init_db()
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
    log_message(req.session_id, "user", q)

    # If this session is in takeover mode, never answer with AI.
    if is_human_mode(req.session_id):
        upsert_support_request(req.session_id, q, req.page_url or "")

        if can_send_booking_email():
            try:
                send_email(
                    OWNER_NOTIFY_EMAIL,
                    "Abood Freediver: Human takeover message",
                    f"""
                    <h3>Human takeover message</h3>
                    <p><b>Session:</b> {req.session_id}</p>
                    <p><b>Page:</b> {req.page_url or ''}</p>
                    <p><b>User message:</b> {q}</p>
                    """,
                )
            except Exception as e:
                print("Email takeover notify failed:", str(e))

        answer = "Thanks — this is the Abood Freediver team. Abood will reply here shortly."
        log_message(req.session_id, "assistant", answer)
        return ChatResponse(answer=answer, needs_human=True, booking_pending=False, booking_next_url=None, sources=[])

    # Human takeover / safety triggers
    if is_medical_or_high_risk(q) or wants_human(q):
        set_human_mode(req.session_id, True)
        upsert_support_request(req.session_id, q, req.page_url or "")

        if can_send_booking_email():
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

        answer = (
            "Thanks — this is the Abood Freediver team. "
            "For safety and accuracy, Abood will reply directly here. "
            "Please share your name and preferred contact (WhatsApp or email)."
        )
        log_message(req.session_id, "assistant", answer)
        return ChatResponse(answer=answer, needs_human=True, booking_pending=False, booking_next_url=None, sources=[])

    latest_booking = get_latest_booking_for_session(req.session_id)

    if latest_booking and looks_like_booking(q):
        status = (latest_booking.get("status") or "").lower()
        details = {}
        try:
            details = json.loads(latest_booking.get("details_json") or "{}")
        except Exception:
            details = {}

        if status == "pending":
            answer = (
                "Your booking request is still pending Abood’s approval.\n\n"
                "If you want to update anything (date/time, number of people, contact), "
                "tell us here and we will forward it."
            )
            log_message(req.session_id, "assistant", answer)
            return ChatResponse(answer=answer, needs_human=True, booking_pending=True, booking_next_url=None, sources=[])

        if status in ("approved", "denied") and wants_new_booking(q):
            details2 = {"question": q, "page_url": req.page_url or "", "history": req.history or []}
            booking_id = create_booking_request(req.session_id, details2)

            if can_send_booking_email():
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

            answer = (
                "I can take your new booking request, but I can’t confirm it until Abood approves.\n\n"
                "Please share:\n"
                "1) Desired date(s) + time window\n"
                "2) Number of people\n"
                "3) Your contact (WhatsApp or email)\n\n"
                "I’ve sent your request to Abood for approval."
            )
            log_message(req.session_id, "assistant", answer)
            return ChatResponse(answer=answer, needs_human=True, booking_pending=True, booking_next_url=None, sources=[])

        if status == "approved":
            lang = detect_language(q, req.history)
            contact_url = build_contact_form_url(details.get("page_url", req.page_url or ""), lang=lang)
            answer = (
                "Approved — please continue the booking on our contact form here:\n"
                f"{contact_url}\n\n"
                "After you submit the form, you can keep chatting here if you have questions."
            )
            log_message(req.session_id, "assistant", answer)
            return ChatResponse(answer=answer, needs_human=False, booking_pending=False, booking_next_url=contact_url, sources=[])

        if status == "denied":
            answer = (
                "Sorry — we couldn’t approve that booking request.\n"
                "If you want, you can try different dates/times.\n"
                "You can also ask any other questions here."
            )
            log_message(req.session_id, "assistant", answer)
            return ChatResponse(answer=answer, needs_human=False, booking_pending=False, booking_next_url=None, sources=[])

    if looks_like_booking(q):
        details = {"question": q, "page_url": req.page_url or "", "history": req.history or []}
        booking_id = create_booking_request(req.session_id, details)

        if can_send_booking_email():
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

        answer = (
            "This is the Abood Freediver team. I can take your booking request, but I can’t confirm it until Abood approves.\n\n"
            "Please share:\n"
            "1) Desired date(s) + time window\n"
            "2) Number of people\n"
            "3) Your contact (WhatsApp or email)\n\n"
            "I’ve sent your request to Abood for approval."
        )
        log_message(req.session_id, "assistant", answer)
        return ChatResponse(answer=answer, needs_human=True, booking_pending=True, booking_next_url=None, sources=[])

    # Retrieval + AI answer
    chunks, conf = await retrieve_site_context(q)

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
    log_message(req.session_id, "assistant", answer)

    out_sources = sources[:4] if chunks and conf >= MIN_CONFIDENCE else []
    for r in web_results[:3]:
        if r.get("link"):
            out_sources.append({"title": r.get("title", "Web source"), "url": r["link"]})

    return ChatResponse(answer=answer, needs_human=False, booking_pending=False, booking_next_url=None, sources=out_sources)


@app.get("/chat/status")
def chat_status(session_id: str = Query(...)):
    return {"messages": get_human_messages(session_id)}


# =========================
# Email-token booking approval/deny
# =========================
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

    add_human_message(
        booking["session_id"],
        f"Approved ✅\nPlease continue the booking on our contact form: {contact_url}\nAfter you submit the form, you can keep chatting here if you have any questions.",
    )
    log_message(booking["session_id"], "human", f"Approved ✅ link shared: {contact_url}")
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
    log_message(booking["session_id"], "human", "Denied booking request (email-token).")
    return {"ok": True, "status": "denied", "booking_id": booking_id}


# =========================
# Admin dashboard + APIs (Basic Auth)
# =========================
@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(creds: HTTPBasicCredentials = Depends(security)):
    require_admin(creds)

    html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Abood Freediver Admin</title>
  <style>
    body{font-family:Arial, sans-serif; margin:20px;}
    h2{margin-top:28px;}
    table{border-collapse:collapse; width:100%;}
    th,td{border:1px solid #ddd; padding:8px; vertical-align:top;}
    th{background:#f5f5f5;}
    textarea{width:100%; height:70px;}
    button{padding:8px 12px; margin-right:6px; cursor:pointer;}
    .row-actions{white-space:nowrap;}
    .muted{color:#666; font-size:12px;}
    pre{margin:0;}
    .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#eee;font-size:12px;margin-left:6px;}
    .modal-backdrop{position:fixed; inset:0; background:rgba(0,0,0,.5); display:none; align-items:center; justify-content:center;}
    .modal{background:#fff; width:min(1000px, 95vw); max-height:90vh; overflow:auto; border-radius:10px; padding:14px;}
    .modal h3{margin:6px 0 12px 0;}
    .chatline{padding:10px; border:1px solid #eee; border-radius:10px; margin-bottom:10px;}
    .chatline .meta{font-size:12px; color:#666; margin-bottom:6px;}
    .chatline.user{background:#f8fbff;}
    .chatline.assistant{background:#fbfff8;}
    .chatline.human{background:#fff8fb;}
    .closebtn{float:right;}

    .sending { opacity: 0.6; pointer-events: none; }
    .small { font-size:12px; color:#666; }
  </style>
</head>
<body>
  <h1>Abood Freediver Admin Dashboard</h1>
  <div class="muted">Protected by Basic Auth (ADMIN_USER / ADMIN_PASS).</div>

  <h2>Pending Bookings</h2>
  <table id="bookingsTbl">
    <thead><tr><th>ID</th><th>Session</th><th>Details</th><th>Updated</th><th>Actions</th></tr></thead>
    <tbody></tbody>
  </table>

  <h2>Open Human Help Requests</h2>
  <table id="supportTbl">
    <thead><tr><th>Session</th><th>Message</th><th>Page</th><th>Updated</th><th>Conversation</th><th>Send Reply</th></tr></thead>
    <tbody></tbody>
  </table>

  <div class="modal-backdrop" id="backdrop">
    <div class="modal">
      <button class="closebtn" type="button" onclick="closeModal()">Close</button>
      <h3>Conversation: <span id="convTitle"></span></h3>
      <div id="convBody"></div>
    </div>
  </div>

<script>
async function apiGet(url){
  const r = await fetch(url, {credentials:"include"});
  if(!r.ok){ throw new Error(await r.text()); }
  return await r.json();
}
async function apiPost(url, body){
  const r = await fetch(url, {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body || {}),
    credentials:"include"
  });
  if(!r.ok){ throw new Error(await r.text()); }
  return await r.json();
}
function fmtTs(ts){
  if(!ts) return "";
  const d = new Date(ts*1000);
  return d.toLocaleString();
}
function esc(s){
  return (s||"").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

/* -----------------------------
   Draft persistence (FIX)
----------------------------- */
function draftKey(sessionId){ return "draft_reply_" + sessionId; }

function saveDraft(sessionId, value){
  try { localStorage.setItem(draftKey(sessionId), value || ""); } catch(e){}
}

function loadDraft(sessionId){
  try { return localStorage.getItem(draftKey(sessionId)) || ""; } catch(e){ return ""; }
}

function clearDraft(sessionId){
  try { localStorage.removeItem(draftKey(sessionId)); } catch(e){}
}

// Capture drafts from existing textareas BEFORE we rebuild the table
function captureDraftsFromDom(){
  document.querySelectorAll("textarea.reply-box[data-session]").forEach((ta) => {
    saveDraft(ta.dataset.session, ta.value || "");
  });
}

// After we rebuild, restore drafts and attach listeners
function wireDraftTextareas(){
  document.querySelectorAll("textarea.reply-box[data-session]").forEach((ta) => {
    const sid = ta.dataset.session;

    // Restore only if empty (do not overwrite live typing)
    if(!ta.value){
      const d = loadDraft(sid);
      if(d) ta.value = d;
    }

    // Save while typing
    ta.addEventListener("input", () => saveDraft(sid, ta.value || ""));
  });
}

/* -----------------------------
   Data loading
----------------------------- */
async function loadBookings(){
  const data = await apiGet("/admin/api/bookings?status=pending");
  const tb = document.querySelector("#bookingsTbl tbody");
  tb.innerHTML = "";
  (data.items || []).forEach(it => {
    let details = it.details_json || "";
    try{ details = JSON.stringify(JSON.parse(details), null, 2); }catch(e){}
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${it.id}</td>
      <td>${esc(it.session_id)}</td>
      <td><pre style="white-space:pre-wrap">${esc(details)}</pre></td>
      <td>${fmtTs(it.updated_at)}</td>
      <td class="row-actions">
        <button type="button" onclick="approveBooking(${it.id})">Approve</button>
        <button type="button" onclick="denyBooking(${it.id})">Deny</button>
      </td>
    `;
    tb.appendChild(tr);
  });
}

async function approveBooking(id){
  await apiPost(`/admin/api/bookings/${id}/approve`, {});
  await loadBookings();
}
async function denyBooking(id){
  await apiPost(`/admin/api/bookings/${id}/deny`, {});
  await loadBookings();
}

async function loadSupport(){
  // IMPORTANT: save drafts before rebuilding table
  captureDraftsFromDom();

  const data = await apiGet("/admin/api/support?status=open");
  const tb = document.querySelector("#supportTbl tbody");
  tb.innerHTML = "";

  (data.items || []).forEach(it => {
    const page = it.page_url || "";
    const sid = it.session_id || "";
    const sidEsc = esc(sid);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${sidEsc}</td>
      <td>${esc(it.user_message)}</td>
      <td>${page ? `<a target="_blank" rel="noopener noreferrer" href="${esc(page)}">${esc(page)}</a>` : ""}</td>
      <td>${fmtTs(it.updated_at)}</td>
      <td><button type="button" onclick="openConversation('${sidEsc}')">View</button></td>
      <td>
        <textarea
          class="reply-box"
          data-session="${sidEsc}"
          placeholder="Type your reply..."
          id="msg_${sidEsc}"
        ></textarea>
        <div style="margin-top:6px;">
          <button type="button" onclick="sendReply('${sidEsc}', false)">Send</button>
          <button type="button" onclick="sendReply('${sidEsc}', true)">Send &amp; Close</button>
          <span class="small" id="status_${sidEsc}"></span>
        </div>
      </td>
    `;
    tb.appendChild(tr);
  });

  // Restore drafts + attach input listeners after table is rendered
  wireDraftTextareas();
}

async function sendReply(sessionId, closeIt){
  const el = document.getElementById("msg_" + sessionId);
  const statusEl = document.getElementById("status_" + sessionId);

  const text = (el && el.value || "").trim();
  if(!text){ alert("Write a message first"); return; }

  // UI lock for this row to avoid double send
  const td = el ? el.closest("td") : null;
  if(td) td.classList.add("sending");
  if(statusEl) statusEl.textContent = "Sending...";

  try{
    await apiPost("/admin/api/support/message", {session_id: sessionId, text, close: closeIt});

    // Clear draft ONLY after successful send
    clearDraft(sessionId);

    if(el) el.value = "";
    if(statusEl) statusEl.textContent = closeIt ? "Sent & closed." : "Sent.";
    await loadSupport();
  }catch(err){
    if(statusEl) statusEl.textContent = "Error sending.";
    alert("Send failed: " + (err && err.message ? err.message : err));
  }finally{
    if(td) td.classList.remove("sending");
  }
}

function closeModal(){
  document.getElementById("backdrop").style.display = "none";
  document.getElementById("convBody").innerHTML = "";
  document.getElementById("convTitle").textContent = "";
}

async function openConversation(sessionId){
  const data = await apiGet(`/admin/api/conversation?session_id=${encodeURIComponent(sessionId)}&limit=300`);
  document.getElementById("convTitle").textContent = sessionId;
  const body = document.getElementById("convBody");
  body.innerHTML = "";

  (data.items || []).forEach(m => {
    const role = (m.role || "").toLowerCase();
    const div = document.createElement("div");
    div.className = "chatline " + (role || "assistant");
    div.innerHTML = `
      <div class="meta"><b>${esc(role)}</b> <span class="pill">${esc(fmtTs(m.ts))}</span></div>
      <div>${esc(m.text || "").replace(/\\n/g, "<br>")}</div>
    `;
    body.appendChild(div);
  });

  document.getElementById("backdrop").style.display = "flex";
}

(async function init(){
  await loadBookings();
  await loadSupport();

  // periodic refresh WITHOUT losing drafts
  setInterval(async ()=>{
    try{
      await loadBookings();
      await loadSupport();
    }catch(e){
      // ignore transient errors
      console.error(e);
    }
  }, 5000);
})();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/admin/api/bookings")
def admin_api_bookings(status: str = "pending", creds: HTTPBasicCredentials = Depends(security)):
    require_admin(creds)
    return {"items": list_booking_requests(status=status)}


@app.post("/admin/api/bookings/{booking_id}/approve")
def admin_api_approve_booking(booking_id: int, creds: HTTPBasicCredentials = Depends(security)):
    require_admin(creds)
    booking = get_booking(booking_id)
    update_booking_status(booking_id, "approved")

    details = {}
    try:
        details = json.loads(booking.get("details_json") or "{}")
    except Exception:
        details = {}

    lang = detect_language(details.get("question", ""), details.get("history") or [])
    contact_url = build_contact_form_url(details.get("page_url", ""), lang=lang)

    msg = f"Approved ✅\nPlease continue the booking on our contact form: {contact_url}\nAfter you submit the form, you can keep chatting here if you have any questions."
    add_human_message(booking["session_id"], msg)
    log_message(booking["session_id"], "human", msg)
    return {"ok": True}


@app.post("/admin/api/bookings/{booking_id}/deny")
def admin_api_deny_booking(booking_id: int, creds: HTTPBasicCredentials = Depends(security)):
    require_admin(creds)
    booking = get_booking(booking_id)
    update_booking_status(booking_id, "denied")

    msg = (
        "Sorry — we can’t approve this booking request right now.\n"
        "If you share alternative dates/times (and your experience level), we can check again.\n"
        "If you have any other questions, you can ask here."
    )
    add_human_message(booking["session_id"], msg)
    log_message(booking["session_id"], "human", msg)
    return {"ok": True}


@app.get("/admin/api/support")
def admin_api_support(status: str = "open", creds: HTTPBasicCredentials = Depends(security)):
    require_admin(creds)
    return {"items": list_support_requests(status=status)}


@app.post("/admin/api/support/message")
def admin_api_support_message(req: AdminMessageRequest, creds: HTTPBasicCredentials = Depends(security)):
    require_admin(creds)

    add_human_message(req.session_id, req.text)
    log_message(req.session_id, "human", req.text)

    if req.close:
        close_support_request(req.session_id)
        set_human_mode(req.session_id, False)

    return {"ok": True}


@app.get("/admin/api/conversation")
def admin_api_conversation(session_id: str = Query(...), limit: int = 300, creds: HTTPBasicCredentials = Depends(security)):
    require_admin(creds)
    return {"items": get_conversation(session_id, limit=limit)}


# =========================
# Index admin endpoints
# =========================
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
