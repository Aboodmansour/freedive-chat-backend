# main.py
from pathlib import Path
from dotenv import load_dotenv
import os
import time
import json
import math
import sqlite3
import secrets
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple

import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bs4 import BeautifulSoup
from itsdangerous import URLSafeSerializer, BadSignature


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
SERPAPI_KEY = os.getenv("SEARCHAPI_KEY", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    # safest for production is explicit origins, but keep your current behavior
    ALLOWED_ORIGINS = ["*"]

ADMIN_TOKEN_SECRET = os.getenv("ADMIN_TOKEN_SECRET", "").strip() or ("CHANGE_ME_" + secrets.token_urlsafe(16))
serializer = URLSafeSerializer(ADMIN_TOKEN_SECRET, salt="booking-approval")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()  # e.g. https://freedive-chat-backend.onrender.com
DB_PATH = os.getenv("DB_PATH", "data.sqlite3")

MAX_URLS = int(os.getenv("MAX_URLS", "150"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "800"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.20"))


# =========================
# OpenAI (async)
# =========================
openai_client = None
if OPENAI_API_KEY:
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def _require_openai():
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")


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

    # booking status: pending -> approved/denied
    # details_json holds: { "question": "...", "page_url": "...", "history": [...],
    #                      "customer_info": {...}, "raw_messages": [...] }
    cur.execute("""
      CREATE TABLE IF NOT EXISTS booking_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        status TEXT NOT NULL,
        details_json TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      )
    """)

    conn.commit()
    conn.close()


def now_ts() -> int:
    return int(time.time())


def ensure_session(session_id: str):
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO sessions(session_id, created_at) VALUES(?,?)",
        (session_id, now_ts()),
    )
    conn.commit()
    conn.close()


def add_human_message(session_id: str, text: str):
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO human_messages(session_id, text, ts) VALUES(?,?,?)",
        (session_id, text, now_ts()),
    )
    conn.commit()
    conn.close()


def get_human_messages(session_id: str) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT text, ts FROM human_messages WHERE session_id=? ORDER BY ts ASC",
        (session_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": "Abood Freediver Team", "text": r["text"], "ts": r["ts"]} for r in rows]


def chunks_count() -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM chunks")
    n = cur.fetchone()["n"]
    conn.close()
    return int(n)


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
        "from": {"email": (os.getenv("SMTP_FROM", "").strip() or to_email)},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }
    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"SendGrid error {r.status_code}: {r.text}")


def safe_send_owner_email(subject: str, html: str):
    if not (OWNER_NOTIFY_EMAIL and SENDGRID_API_KEY):
        return
    try:
        send_email(OWNER_NOTIFY_EMAIL, subject, html)
    except Exception:
        # do not break user flow
        pass


# =========================
# Utilities (text + retrieval)
# =========================
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
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
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
        "book", "booking", "reserve", "reservation", "schedule",
        "fun dive", "course", "training", "session", "availability",
        "price", "join", "padi"
    ]
    return any(t in ql for t in triggers)


# =========================
# OpenAI helpers
# =========================
async def embed_text(text: str) -> List[float]:
    _require_openai()
    resp = await openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text[:6000],
    )
    return list(resp.data[0].embedding)


async def generate_answer(system: str, user: str) -> str:
    """
    Use chat.completions for gpt-4o-mini (supports temperature).
    """
    _require_openai()
    resp = await openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# =========================
# Retrieval
# =========================
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
            out.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return out
    except Exception:
        return []


# =========================
# Booking flow
# =========================
REQUIRED_BOOKING_FIELDS = [
    "full_name",
    "age",
    "whatsapp_or_phone",
    "email",
    "session_type",  # training | fun_dive
    "date_time",
    "people_count",
    "certification_status",  # certified | not_certified
    "certification_copy_required",  # yes/no
    "id_copy_required",  # yes/no
    "height_cm",
    "weight_kg",
    "feet_size",
    "need_equipment",  # yes/no
    "underwater_photography",  # none | action | pro
    "payment_method",  # cash_arrival | online
]

FUN_DIVE_EXTRA_FIELDS = [
    "preferred_dive_site",
]

TRAINING_EXTRA_FIELDS = [
    "training_focus",
]


def create_booking_request(session_id: str, details: Dict[str, Any]) -> int:
    conn = db()
    cur = conn.cursor()
    ts = now_ts()
    cur.execute(
        "INSERT INTO booking_requests(session_id,status,details_json,created_at,updated_at) VALUES(?,?,?,?,?)",
        (session_id, "pending", json.dumps(details), ts, ts),
    )
    conn.commit()
    booking_id = int(cur.lastrowid)
    conn.close()
    return booking_id


def update_booking(booking_id: int, status: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM booking_requests WHERE id=?", (booking_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Booking not found")

    new_status = status or row["status"]
    current_details = json.loads(row["details_json"] or "{}")
    if details:
        # shallow merge
        current_details.update(details)

    cur.execute(
        "UPDATE booking_requests SET status=?, details_json=?, updated_at=? WHERE id=?",
        (new_status, json.dumps(current_details), now_ts(), booking_id),
    )
    conn.commit()
    conn.close()


def get_booking(booking_id: int) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM booking_requests WHERE id=?", (booking_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    d = dict(row)
    try:
        d["details"] = json.loads(d.get("details_json") or "{}")
    except Exception:
        d["details"] = {}
    return d


def get_latest_booking_for_session(session_id: str) -> Optional[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT * FROM booking_requests WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d["details"] = json.loads(d.get("details_json") or "{}")
    except Exception:
        d["details"] = {}
    return d


def make_approval_links(base_url: str, booking_id: int) -> Tuple[str, str]:
    token = serializer.dumps({"booking_id": booking_id})
    approve = f"{base_url}/admin/booking/approve?token={token}"
    deny = f"{base_url}/admin/booking/deny?token={token}"
    return approve, deny


def booking_missing_fields(details: Dict[str, Any]) -> List[str]:
    info = (details or {}).get("customer_info") or {}
    missing = [k for k in REQUIRED_BOOKING_FIELDS if not info.get(k)]
    st = (info.get("session_type") or "").lower().strip()

    if st == "fun_dive":
        missing += [k for k in FUN_DIVE_EXTRA_FIELDS if not info.get(k)]
        # for fun dive, certification copy is mandatory
        if info.get("certification_status") != "certified":
            # if not certified, they should not book fun dive; keep it missing as signal
            if "certification_status" not in missing:
                missing.append("certification_status")
        if not info.get("certification_copy_required"):
            missing.append("certification_copy_required")

    if st == "training":
        missing += [k for k in TRAINING_EXTRA_FIELDS if not info.get(k)]

    # de-dup preserve order
    seen = set()
    out = []
    for m in missing:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def simple_extract_customer_info(text: str) -> Dict[str, Any]:
    """
    Minimal extraction (no LLM). This keeps your system stable and still useful.
    Customer can reply using the template; we store raw and extract what we can.
    """
    t = (text or "").strip()

    phone = None
    m = re.search(r"(\+\d{7,15}|\b\d{9,15}\b)", t)
    if m:
        phone = m.group(1)

    email = None
    m = re.search(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", t, flags=re.I)
    if m:
        email = m.group(1)

    # rough numbers
    age = None
    m = re.search(r"\bage\s*[:=]\s*(\d{1,2})\b", t, flags=re.I)
    if m:
        age = int(m.group(1))

    people = None
    m = re.search(r"(?:people|persons|divers|guests)\s*[:=]\s*(\d{1,2})\b", t, flags=re.I)
    if m:
        people = int(m.group(1))

    return {
        "whatsapp_or_phone": phone,
        "email": email,
        "age": age,
        "people_count": people,
    }


BOOKING_TEMPLATE = """Please reply with the details below (copy/paste and fill):

Full name:
Age:
WhatsApp/Phone:
Email:
Session type (training or fun_dive):
Preferred date/time:
Number of people:

Certified? (certified or not_certified):
If certified: upload/send certification copy (mandatory for fun dive).
If not certified: upload/send ID copy.

Height (cm):
Weight (kg):
Feet size:
Need equipment? (yes/no):

Underwater photography add-on? (none/action/pro):
Payment (cash_arrival or online):

If fun dive: preferred dive site (optional):
If training: what do you want to focus on? (equalization / duck dive / relaxation / depth / technique / other):
"""


# =========================
# FastAPI
# =========================
app = FastAPI(title="Freedive Chat Backend")

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
    sources: Optional[List[Dict[str, str]]] = None  # keep, but frontend can ignore


@app.on_event("startup")
async def startup():
    init_db()


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/chat/status")
def chat_status(session_id: str = Query(...)):
    return {"messages": get_human_messages(session_id)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question or not req.session_id:
        raise HTTPException(status_code=400, detail="question and session_id required")

    ensure_session(req.session_id)
    q = (req.question or "").strip()

    # If there is an approved booking for this session, collect details and notify owner.
    latest = get_latest_booking_for_session(req.session_id)
    if latest and latest["status"] == "approved":
        details = latest.get("details") or {}
        details.setdefault("raw_messages", [])
        details["raw_messages"].append({"ts": now_ts(), "text": q})

        details.setdefault("customer_info", {})
        details["customer_info"].update(simple_extract_customer_info(q))

        missing = booking_missing_fields(details)

        update_booking(int(latest["id"]), details=details)

        # email owner every time, but include missing list (so you can follow up)
        html = f"""
        <h3>Approved booking details update</h3>
        <p><b>Booking ID:</b> {latest["id"]}</p>
        <p><b>Session:</b> {req.session_id}</p>
        <p><b>Page:</b> {escape_html(req.page_url or "")}</p>
        <p><b>Latest customer message:</b></p>
        <pre>{escape_html(q)}</pre>
        <p><b>Extracted fields (partial):</b></p>
        <pre>{escape_html(json.dumps(details.get("customer_info", {}), indent=2))}</pre>
        <p><b>Missing required fields:</b></p>
        <pre>{escape_html(", ".join(missing) if missing else "None (complete)")}</pre>
        """
        safe_send_owner_email("Approved booking: details update", html)

        if missing:
            answer = (
                "This is the Abood Freediver Team. Thanks — your booking is approved.\n\n"
                "To finalize, please provide the remaining details using this template:\n\n"
                f"{BOOKING_TEMPLATE}"
            )
            return ChatResponse(answer=answer, needs_human=True, booking_pending=True, sources=[])

        # complete
        answer = (
            "This is the Abood Freediver Team. Perfect — we have all required details.\n\n"
            "We will confirm the final schedule, meeting point, and payment instructions shortly."
        )
        return ChatResponse(answer=answer, needs_human=True, booking_pending=False, sources=[])

    # If booking intent: create pending booking, notify owner with approve/deny links
    if looks_like_booking(q):
        booking_details = {
            "question": q,
            "page_url": req.page_url or "",
            "history": req.history or [],
            "customer_info": {},
            "raw_messages": [{"ts": now_ts(), "text": q}],
        }
        booking_id = create_booking_request(req.session_id, booking_details)

        approve, deny = ("", "")
        if PUBLIC_BASE_URL:
            approve, deny = make_approval_links(PUBLIC_BASE_URL, booking_id)

        safe_send_owner_email(
            "Booking request pending approval",
            f"""
            <h3>Booking request pending approval</h3>
            <p><b>Booking ID:</b> {booking_id}</p>
            <p><b>Session:</b> {req.session_id}</p>
            <p><b>Page:</b> {escape_html(req.page_url or "")}</p>
            <p><b>User message:</b></p>
            <pre>{escape_html(q)}</pre>
            <p><b>Approve:</b> <a href="{approve}">{approve}</a></p>
            <p><b>Deny:</b> <a href="{deny}">{deny}</a></p>
            """,
        )

        return ChatResponse(
            answer=(
                "This is the Abood Freediver Team. I can take your booking request, "
                "but I can only confirm after instructor approval.\n\n"
                "Please share these details (copy/paste and fill):\n\n"
                f"{BOOKING_TEMPLATE}"
            ),
            needs_human=True,
            booking_pending=True,
            sources=[],
        )

    # Normal Q&A (RAG)
    chunks, conf = await retrieve_site_context(q)

    system = (
        "You are Aqua from the Abood Freediver Team (Aqaba).\n"
        "Always speak as the Abood Freediver Team.\n"
        "Rules:\n"
        "- Prefer answering using SITE_CONTEXT.\n"
        "- If SITE_CONTEXT is insufficient, you may use WEB_CONTEXT if provided.\n"
        "- Never confirm a booking unless it is already approved in the system.\n"
        "- Keep answers concise, accurate, and safety-conscious.\n"
        "- Do not mention internal embeddings, databases, or implementation details.\n"
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
        "- If you use website info, you may internally rely on it, but do not show 'Sources' unless asked.\n"
        "- If info is missing, ask a short follow-up question.\n"
    )

    answer = await generate_answer(system, prompt)

    return ChatResponse(answer=answer, needs_human=False, booking_pending=False, sources=sources[:4])


def escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


@app.get("/admin/booking/approve")
def approve_booking(token: str = Query(...)):
    try:
        data = serializer.loads(token)
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid token")

    booking_id = int(data["booking_id"])
    booking = get_booking(booking_id)

    update_booking(booking_id, status="approved")

    # user-facing confirmation + mandatory fields request
    add_human_message(
        booking["session_id"],
        "Abood Freediver Team: Approved. Please reply with the required details to finalize.\n\n"
        + BOOKING_TEMPLATE
    )

    # owner notification
    details = booking.get("details") or {}
    safe_send_owner_email(
        "Booking approved",
        f"""
        <h3>Booking approved</h3>
        <p><b>Booking ID:</b> {booking_id}</p>
        <p><b>Session:</b> {escape_html(booking.get("session_id",""))}</p>
        <p><b>Original request:</b></p>
        <pre>{escape_html((details.get("question") or ""))}</pre>
        <p>The customer will now be asked for mandatory info.</p>
        """,
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

    update_booking(booking_id, status="denied")

    add_human_message(
        booking["session_id"],
        "Abood Freediver Team: We can’t confirm that booking request yet. "
        "Please share alternative dates/times and your experience level."
    )

    safe_send_owner_email(
        "Booking denied",
        f"""
        <h3>Booking denied</h3>
        <p><b>Booking ID:</b> {booking_id}</p>
        <p><b>Session:</b> {escape_html(booking.get("session_id",""))}</p>
        """,
    )

    return {"ok": True, "status": "denied", "booking_id": booking_id}
