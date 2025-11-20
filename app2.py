# ===============================
# 🤖 TLC Training Chatbot — Local Full App
# ===============================
import os, re, textwrap
import numpy as np
from datetime import datetime
import gradio as gr
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# =====================================================
# 🩹 PATCH: Fix "argument of type bool is not iterable"
# =====================================================
import gradio_client.utils as _gcu

_json_schema_to_python_type_orig = _gcu.json_schema_to_python_type
def _json_schema_to_python_type_safe(schema):
    if isinstance(schema, bool):
        return "any" if schema else "never"
    try:
        return _json_schema_to_python_type_orig(schema)
    except Exception:
        return "any"
_gcu.json_schema_to_python_type = _json_schema_to_python_type_safe

_get_type_orig = _gcu.get_type
def _get_type_safe(schema):
    if not isinstance(schema, dict):
        return "any"
    try:
        return _get_type_orig(schema)
    except Exception:
        return "any"
_gcu.get_type = _get_type_safe

# =====================================================
# ⚙️ CONFIGURATION
# =====================================================
ORG_NAME = "Toyota Learning Center (TLC) – TMMIN"
CONTACT_EMAIL = "tlc@toyota.co.id"
REG_FORM_URL = "https://forms.gle/your-form-id"   # ganti sesuai form kamu
WHATSAPP_LINK = "https://wa.me/6281234567890?text=Halo%20TLC%2C%20saya%20ingin%20daftar%20training"
LOCATIONS = ["TLC Sunter 2 – Zenix 2", "TLC Karawang", "Online (MS Teams)", "On-site (Supplier)"]

# Intent routing knobs — tweak to adjust how strict the classifier vs semantic router should be
INTENT_CONFIDENCE_THRESHOLD = 0.5  # if logistic classifier confidence >= this, trust it
SEMANTIC_INTENT_THRESHOLD = 0.6    # otherwise require at least this cosine similarity for semantic routing

PRICING_KEYWORDS = ["harga", "biaya", "cost", "fee", "tarif"]
EXTERNAL_KEYWORDS = [
    "in-house",
    "in house",
    "inhouse",
    "di luar pabrik",
    "di luar plant",
    "diadakan di luar",
    "onsite",
    "on-site",
    "di lokasi kami",
    "bisa external",
    "di luar site",
    "bisa diadakan di luar",
]

# =====================================================
# 📚 SAMPLE DATA
# =====================================================
courses = pd.DataFrame([
    {
        "code": "JKK-SV-101",
        "title": "Jikotei Kanketsu for Supervisor",
        "audience": "Supervisor / Section Head (Supplier & Internal)",
        "format": "Offline",
        "duration_days": 1,
        "next_runs": "2025-11-17;2025-12-03",
        "location": "TLC Sunter 2 – Zenix 2",
        "price_idr": 1850000,
        "prereq": "Dasar PDCA",
        "description": "Prinsip JKK, PDCA, standard work audit, problem finding di lini produksi supplier."
    },
    {
        "code": "TCLASS-FOUND-201",
        "title": "TClass LMS Foundation",
        "audience": "Admin HRD Supplier, Training PIC",
        "format": "Online",
        "duration_days": 0.5,
        "next_runs": "2025-11-20;2025-12-10",
        "location": "Online (MS Teams)",
        "price_idr": 950000,
        "prereq": "-",
        "description": "Pengantar TClass: membuat kelas, import peserta, tracking jam training, laporan compliance."
    },
    {
        "code": "QCC-LEAD-301",
        "title": "QCC Leadership & Facilitation",
        "audience": "Team Leader / Champion QCC",
        "format": "Offline",
        "duration_days": 1,
        "next_runs": "2025-12-05",
        "location": "TLC Karawang",
        "price_idr": 2250000,
        "prereq": "Basic Quality Tools",
        "description": "Memandu tim QCC, fishbone, 7 tools, storytelling improvement, persiapan presentasi juri."
    },
])
faqs = pd.DataFrame([
    {"q": "Apakah program TLC terbuka untuk supplier?",
     "a": "Ya, program tertentu dibuka untuk supplier. Lihat katalog & jadwal atau hubungi kami untuk batch khusus."},
    {"q": "Metode pembayaran?",
     "a": "Transfer bank (invoice) atau e-invoice vendor terdaftar. Konfirmasi ke " + CONTACT_EMAIL + "."},
    {"q": "Lokasi training offline?",
     "a": f"Utamanya {LOCATIONS[0]} dan {LOCATIONS[1]}. Bisa on-site di pabrik supplier."},
    {"q": "Kebijakan pembatalan?",
     "a": "Pembatalan ≤ D-5: refund penuh; D-4 s.d D-2: 50 %; D-1/no-show: tidak refund (bisa reschedule bila slot tersedia)."},
    {"q": "Apakah TLC bisa in-house di lokasi perusahaan?",
     "a": "Bisa, kami dapat mengadakan pelatihan on-site/in-house di pabrik atau kantor Anda sesuai jadwal yang disepakati."},
    {"q": "Berapa minimal peserta untuk in-house training?",
     "a": "Minimal kuota biasanya ±15 peserta (dapat disesuaikan). Mohon info estimasi peserta untuk proposal."},
    {"q": "Bagaimana alur permintaan in-house training?",
     "a": "Umumnya: kirim kebutuhan/topik & jumlah peserta → klarifikasi tujuan & profil peserta → proposal & jadwal → delivery."},
    {"q": "Apa faktor yang mempengaruhi harga in-house training?",
     "a": "Jumlah peserta, lokasi, materi/level kustomisasi, serta kebutuhan alat/praktik di lapangan."},
    {"q": "Bisakah materi disesuaikan dengan proses perusahaan?",
     "a": "Bisa. Kami dapat menyesuaikan studi kasus/contoh dengan proses dan isu di lini Anda."},
])

# =====================================================
# 🔍 RAG INDEX
# =====================================================
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_corpus():
    docs, meta = [], []
    for _, r in courses.iterrows():
        docs.append(
            f"[COURSE] {r.code} {r.title} | {r.audience} | {r.format}, {r.duration_days} hari | "
            f"Jadwal: {r.next_runs} | Lokasi: {r.location} | Harga Rp{int(r.price_idr):,} | {r.description}"
        )
        meta.append({"type":"course","code":r.code})
    for i, r in faqs.iterrows():
        docs.append(f"[FAQ] Q: {r.q} A: {r.a}")
        meta.append({"type":"faq","id":i})
    return docs, meta

docs, meta = build_corpus()
emb = embed_model.encode(docs, normalize_embeddings=True).astype("float32")
_nn = NearestNeighbors(metric="cosine").fit(emb)

def rag_search(query, k=3):
    qv = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    dists, ids = _nn.kneighbors(qv, n_neighbors=min(k, len(docs)))
    out = []
    for d, i in zip(dists[0], ids[0]):
        out.append((1.0 - float(d), docs[int(i)], meta[int(i)]))
    return out


def summarize_rag_results(rag_results, limit=3):
    """Return human-friendly bullet snippets for the top RAG hits."""
    snippets, titles = [], []
    for score, _, meta_info in rag_results[:limit]:
        if meta_info["type"] == "course":
            r = courses[courses.code == meta_info["code"]].iloc[0]
            snippets.append(
                f"• {r.code} – {r.title}: {r.format}, {r.duration_days} hari. "
                f"Jadwal {r.next_runs} @ {r.location}. Harga Rp{int(r.price_idr):,}."
            )
            titles.append(f"{r.code} – {r.title}")
        else:
            faq_row = faqs.iloc[meta_info["id"]]
            snippets.append(f"• {faq_row.q} → {faq_row.a}")
            titles.append(faq_row.q)
    return snippets, titles

# =====================================================
# 🧠 INTENT CLASSIFIER + SLOT EXTRACTOR
# =====================================================
train = [
    # Catalog / list of programs
    ("Lihat katalog pelatihan", "catalog"),
    ("Tunjukkan semua kursus yang ada", "catalog"),
    ("Ada kelas apa saja di TLC?", "catalog"),
    # Schedule / when
    ("Jadwal JKK kapan", "schedule"),
    ("kapan batch terdekat Jikotei", "schedule"),
    ("bulan depan ada kelas apa", "schedule"),
    # Pricing
    ("Harga TClass berapa", "pricing"),
    ("berapa biaya ikut QCC", "pricing"),
    ("fee training TCLASS", "pricing"),
    # Registration / how to register
    ("Saya mau daftar 5 peserta", "registration"),
    ("Cara daftar training", "registration"),
    ("tolong proses pendaftaran untuk tim kami", "registration"),
    # Custom / in-house
    ("Bisa training di pabrik kami", "custom"),
    ("bisa inhouse di Karawang", "custom"),
    # External / in-house request
    ("Kami ingin ajukan in-house training untuk perusahaan", "external_training_request"),
    ("Bisakah TLC datang ke plant kami untuk JKK?", "external_training_request"),
    ("TLC bisa nggak adain pelatihan di kantor kami?", "external_training_request"),
    ("We need onsite training for our company", "external_training_request"),
    ("Interested in in-house program for 20 pax", "external_training_request"),
    # Policy / cancellation
    ("Kebijakan pembatalan", "policy"),
    ("kalau cancel apa bisa refund", "policy"),
    # Contact
    ("Hubungi siapa", "contact"),
    ("ada nomor WhatsApp", "contact"),
]
X = [x for x, _ in train]
y = [y for _, y in train]
intent_clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("logreg", LogisticRegression(max_iter=1000))
]).fit(X, y)

INTENT_PROFILES = [
    {
        "label": "catalog",
        "examples": ["katalog", "list kursus", "program tersedia", "training apa saja"],
        "description": "Meminta daftar program/kursus yang ada di TLC"
    },
    {
        "label": "schedule",
        "examples": ["jadwal", "kapan kelas", "tanggal batch", "batch berikut"],
        "description": "Menanyakan jadwal atau tanggal pelatihan"
    },
    {
        "label": "pricing",
        "examples": ["harga", "biaya", "tarif", "fee"],
        "description": "Menanyakan biaya / harga pelatihan tertentu"
    },
    {
        "label": "registration",
        "examples": ["daftar", "registrasi", "enroll", "mau ikut"],
        "description": "Cara mendaftar atau permintaan pendaftaran peserta"
    },
    {
        "label": "custom",
        "examples": ["in-house", "onsite", "bawa ke pabrik", "private class"],
        "description": "Meminta pelatihan khusus/in-house di lokasi peserta"
    },
    {
        "label": "external_training_request",
        "examples": ["in-house training", "onsite di pabrik", "datang ke kantor kami", "inhouse untuk company", "request external training"],
        "description": "Permintaan training eksternal/in-house untuk perusahaan"
    },
    {
        "label": "policy",
        "examples": ["batal", "refund", "pembatalan", "cancel"],
        "description": "Kebijakan pembatalan / pembayaran"
    },
    {
        "label": "contact",
        "examples": ["kontak", "hubungi", "nomor wa", "email"],
        "description": "Meminta detail kontak TLC"
    },
]
def build_intent_vectors():
    vectors = {}
    for profile in INTENT_PROFILES:
        label = profile["label"]
        seeds = profile["examples"] + [profile["description"]]
        emb_matrix = embed_model.encode(seeds, normalize_embeddings=True).astype("float32")
        avg_vec = np.mean(emb_matrix, axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm:
            avg_vec = (avg_vec / norm).astype("float32")
        vectors[label] = avg_vec
    return vectors

intent_vectors = build_intent_vectors()

def semantic_intent(text):
    qv = embed_model.encode([text], normalize_embeddings=True).astype("float32")[0]
    best_intent, best_score = None, -1.0
    for intent, vec in intent_vectors.items():
        score = float(np.dot(qv, vec))
        if score > best_score:
            best_intent, best_score = intent, score
    return best_intent, best_score

DATE_RX = re.compile(r"(20\d{2}-\d{2}-\d{2})", re.I)
PAX_RX  = re.compile(r"(\d+)\s*(pax|orang|peserta|people)", re.I)
BARE_PAX_RX = re.compile(r"^\s*(\d{1,3})\s*$")
COURSE_RX = re.compile(r"(JKK[-\s]?\w+|TCLASS[-\s]?\w+|QCC[-\s]?\w+)", re.I)
COMPANY_RX = re.compile(r"(PT\s+[A-Za-z0-9.&()\-\s]+)", re.I)
LOCATION_RX = re.compile(
    r"(karawang|sunter|jakarta|bandung|bekasi|cikarang|purwakarta|cibitung|cikande|depok|bogor|tangerang|semarang|surabaya|bali|yogyakarta|jogja|medan|makassar|batam|balikpapan|samarinda)",
    re.I,
)

def extract_slots(text):
    s = {}
    if m:=PAX_RX.search(text):
        s["pax"]=int(m.group(1))
    elif isinstance(text, str) and BARE_PAX_RX.match(text.strip()):
        s["pax"] = int(text.strip())
    if m:=COURSE_RX.search(text):
        s["course"]=m.group(1).upper().replace(" ","")
    if m:=COMPANY_RX.search(text):
        s["company"]=m.group(1).strip()
    if m:=DATE_RX.search(text):
        s["date"]=m.group(1)
    if m:=LOCATION_RX.search(text):
        s["location"] = m.group(1).title().strip()
    return s

SESSION_STATE_TEMPLATE = {
    "current_intent": None,
    "current_course_code": None,
    "current_course_title": None,
    "participants": None,
    "preferred_dates": None,
    "company_name": None,
    "location": None,
    "mode": None,
}

def init_session_state():
    return SESSION_STATE_TEMPLATE.copy()

def ensure_session_state(state):
    base = init_session_state()
    if isinstance(state, dict):
        for key in base:
            if state.get(key) is not None:
                base[key] = state[key]
    return base

def match_course_reference(text):
    if not text:
        return None
    text_lower = text.lower()
    slots = extract_slots(text)
    if slots.get("course"):
        hint = slots["course"].split("-")[0]
        match = courses[courses.code.str.contains(hint, case=False, regex=False)]
        if not match.empty:
            row = match.iloc[0]
            return {"code": row.code, "title": row.title}
    for _, row in courses.iterrows():
        if row.title.lower() in text_lower:
            return {"code": row.code, "title": row.title}
    return None

def append_unique_date(state, date_str):
    if not date_str:
        return
    if state.get("preferred_dates") is None:
        state["preferred_dates"] = []
    if date_str not in state["preferred_dates"]:
        state["preferred_dates"].append(date_str)

def apply_context_rules(text, intent, debug, state, slots):
    normalized = text.lower()
    clf_conf = debug.get("clf_confidence") or 0.0
    low_conf = (intent is None) or (clf_conf < INTENT_CONFIDENCE_THRESHOLD)
    chosen = intent
    if low_conf and state.get("current_intent") in {"catalog", "registration"}:
        if any(word in normalized for word in PRICING_KEYWORDS):
            chosen = "pricing"
    if low_conf and state.get("current_intent") == "external_training_request":
        if any(word in normalized for word in EXTERNAL_KEYWORDS):
            chosen = "external_training_request"
    if any(word in normalized for word in EXTERNAL_KEYWORDS):
        if state.get("current_course_code") or state.get("current_course_title") or slots.get("course"):
            chosen = "external_training_request"
            state["mode"] = "external"
    if low_conf and (slots.get("location") or state.get("location")):
        if state.get("current_course_code") or state.get("current_course_title") or slots.get("course"):
            chosen = "external_training_request"
            state["mode"] = state.get("mode") or "external"
    if state.get("current_intent") in {"registration", "custom", "external_training_request"} and slots.get("pax"):
        if chosen is None:
            chosen = state.get("current_intent")
    return chosen

def update_state_from_slots(state, slots, course_match=None):
    if not state:
        return
    if course_match:
        state["current_course_code"] = course_match.get("code")
        state["current_course_title"] = course_match.get("title")
    elif slots.get("course"):
        hint = slots["course"].split("-")[0]
        match = courses[courses.code.str.contains(hint, case=False, regex=False)]
        if not match.empty:
            row = match.iloc[0]
            state["current_course_code"] = row.code
            state["current_course_title"] = row.title
    if slots.get("pax"):
        state["participants"] = slots["pax"]
    if slots.get("company"):
        state["company_name"] = slots["company"]
    if slots.get("location"):
        state["location"] = slots["location"]
    if slots.get("date"):
        append_unique_date(state, slots["date"])


def detect_missing_slots(slots, state):
    missing = []
    if not (slots.get("course") or state.get("current_course_code") or state.get("current_course_title")):
        missing.append("course")
    if not (slots.get("company") or state.get("company_name")):
        missing.append("company_name")
    if not (slots.get("pax") or state.get("participants")):
        missing.append("participants")
    if not (slots.get("location") or state.get("location")):
        missing.append("location")
    if not (slots.get("date") or (state.get("preferred_dates") and len(state.get("preferred_dates")) > 0)):
        missing.append("preferred_dates")
    return missing

# =====================================================
# 💬 HANDLERS
# =====================================================
def handle_catalog():
    out = "\n".join(f"- **{r.code} – {r.title}** ({r.format}, {r.duration_days} hari) @ {r.location} • Rp{int(r.price_idr):,}"
                    for _, r in courses.iterrows())
    return "📘 Program yang tersedia:\n" + out

def handle_schedule(text, session_state=None, slots=None):
    slots = slots or extract_slots(text)
    course = slots.get("course")
    if not course and session_state:
        course = session_state.get("current_course_code")
    if course:
        match = courses[courses.code.str.contains(course.split("-")[0], case=False, regex=False)]
    else:
        match = courses
    msg = ["📅 Jadwal:"]
    for _, r in match.iterrows():
        msg.append(f"- {r.code}: {r.next_runs} @ {r.location}")
    return "\n".join(msg)

def handle_pricing(text, session_state=None, slots=None):
    slots = slots or extract_slots(text)
    c = slots.get("course")
    if not c and session_state:
        c = session_state.get("current_course_code")
    if c:
        match = courses[courses.code.str.contains(c.split("-")[0], case=False, regex=False)]
        if match.empty:
            return "Kode tidak ditemukan."
        r = match.iloc[0]
        pax_note = ""
        if session_state and session_state.get("participants"):
            pax_note = f" untuk {session_state['participants']} peserta"
        return f"Harga {r.code} – {r.title}{pax_note}: Rp{int(r.price_idr):,} / peserta."
    if session_state and session_state.get("current_course_title"):
        return (f"Untuk {session_state['current_course_code']} – {session_state['current_course_title']}, "
                "mohon info kode atau jumlah peserta agar kami hitung tepat.")
    return "Sebutkan kode pelatihan (contoh: JKK-SV-101) untuk info harga."

def handle_registration(text, session_state=None, slots=None):
    slots = slots or extract_slots(text)
    comp   = slots.get("company") or (session_state.get("company_name") if session_state else None)
    course = slots.get("course") or (session_state.get("current_course_code") if session_state else None)
    pax    = slots.get("pax") or (session_state.get("participants") if session_state else None)
    # Build ringkas untuk header
    hdr = []
    if comp:   hdr.append(f"• Perusahaan: {comp}")
    if course:
        title = session_state.get("current_course_title") if session_state else None
        suffix = f" – {title}" if title and title not in course else ""
        hdr.append(f"• Kursus: {course}{suffix}")
    if pax:    hdr.append(f"• Jumlah Peserta: {pax}")
    header = ("\n".join(hdr) + "\n") if hdr else ""

    return (
f"{header}**Cara daftar training di TLC:**\n\n"
f"**Opsi A — Public Class (jadwal umum)**\n"
f"1) Isi Google Form: {REG_FORM_URL}\n"
f"   Siapkan data: Perusahaan, Nama peserta, Email, No. HP/WA, Kode kursus (mis. JKK-SV-101), Tanggal batch pilihan, Jumlah peserta.\n"
f"2) Tim kami kirim konfirmasi seat ≤ 1×24 jam kerja.\n"
f"3) Terima quotation/invoice (PO jika perlu) → lakukan pembayaran.\n"
f"4) Dapat email konfirmasi + undangan (lokasi/Teams link).\n\n"
f"**Opsi B — In-House / On-Site (custom untuk supplier)**\n"
f"1) Kirim detail ke email {CONTACT_EMAIL} atau WA {WHATSAPP_LINK}:\n"
f"   Perusahaan, Tujuan, Profil & jumlah peserta, Kode/materi, Tanggal target, Lokasi (TLC/On-site).\n"
f"2) Kami kirim proposal & jadwal ≤ 2×24 jam kerja.\n"
f"3) Konfirmasi + PO/invoice → jadwal fix.\n\n"
f"**Catatan:** Pembatalan ≤ D-5: refund penuh; D-4–D-2: 50%; D-1/no-show: tidak refund (bisa reschedule jika ada slot).\n\n"
f"**Contoh pesan WA:**\n"
f"> Halo TLC, kami ingin mendaftar public class.\n"
f"> Perusahaan: {comp or 'PT ______'}\n"
f"> Kursus: {course or 'JKK-SV-101'}\n"
f"> Jumlah: {pax or '___'} peserta\n"
f"> Batch: { (session_state.get('preferred_dates')[0] if session_state and session_state.get('preferred_dates') else '2025-12-03') }\n"
f"> Kontak: 08xx-xxxx\n"
    )

def handle_custom(text, session_state=None, slots=None):
    slots = slots or extract_slots(text)
    company = slots.get("company") or (session_state.get("company_name") if session_state else None)
    course = session_state.get("current_course_code") if session_state else None
    title = session_state.get("current_course_title") if session_state else None
    mode = session_state.get("mode") if session_state else None
    header = ""
    if course:
        header = f"Fokus materi: {course} – {title or ''}. "
    mode_line = "Mode: external/in-house" if not mode else f"Mode: {mode.title()}"
    return textwrap.dedent(f"""
    {header}In-house bisa untuk perusahaan {company or 'Anda'}.
    • {mode_line}
    • Kirim detail (tujuan, jumlah peserta, lokasi, tanggal target) ke {CONTACT_EMAIL} atau WA {WHATSAPP_LINK}.
    • Minimal kuota ±15 peserta, jadwal fleksibel 2–3 minggu dari konfirmasi.
    """).strip()


def handle_external_training_request(text, session_state=None, slots=None):
    """Echo detected slots for in-house/external training and ask for gaps."""
    slots = slots or extract_slots(text)
    state = ensure_session_state(session_state)
    state["mode"] = state.get("mode") or "external"

    course = slots.get("course") or state.get("current_course_code") or state.get("current_course_title") or "(kode/topik?)"
    course_title = state.get("current_course_title")
    course_line = f"• Kursus/topik: {course}"
    if course_title and course_title not in course:
        course_line += f" – {course_title}"

    company = slots.get("company") or state.get("company_name") or "(nama perusahaan?)"
    pax = slots.get("pax") or state.get("participants") or "(jumlah peserta?)"
    loc = slots.get("location") or state.get("location") or "(lokasi/kota?)"
    dates = slots.get("date") or ", ".join(state.get("preferred_dates") or []) or "(waktu target?)"

    missing = detect_missing_slots(slots, state)
    prompts = []
    if "company_name" in missing:
        prompts.append("Nama perusahaan?")
    if "participants" in missing:
        prompts.append("Estimasi jumlah peserta?")
    if "location" in missing:
        prompts.append("Lokasi training (pabrik/kantor/hotel) dan kota?")
    if "preferred_dates" in missing:
        prompts.append("Tanggal target atau bulan rencana?")
    if "course" in missing:
        prompts.append("Topik atau kode kursus yang diinginkan?")

    lines = [
        "Siap, kami catat permintaan in-house / on-site:",
        f"• Perusahaan: {company}",
        course_line,
        f"• Perkiraan peserta: {pax}",
        f"• Lokasi/area: {loc}",
        f"• Waktu target: {dates}",
        "Materi dapat disesuaikan dengan proses dan studi kasus Anda.",
    ]
    if prompts:
        lines.append("Mohon info tambahan agar proposalnya sesuai:")
        for q in prompts:
            lines.append(f"- {q}")
    else:
        lines.append(f"Tim TLC akan follow-up via email/WA ({CONTACT_EMAIL} / {WHATSAPP_LINK}).")

    return "\n".join(lines)

def handle_policy(): 
    return faqs.iloc[3].a

def handle_contact():
    return f"Hubungi kami di {CONTACT_EMAIL} atau WA {WHATSAPP_LINK}."

def generate_fallback_response(user_text, rag_results, detected_slots, missing_slots):
    """RAG-powered fallback that also nudges for missing info."""
    snippets, titles = summarize_rag_results(rag_results, limit=3)
    topic_hint = titles[0] if titles else "topik pelatihan"

    followups = []
    if "course" in missing_slots:
        followups.append("Kode atau judul kursus yang dimaksud?")
    if "company_name" in missing_slots:
        followups.append("Nama perusahaan Anda?")
    if "participants" in missing_slots:
        followups.append("Estimasi jumlah peserta?")
    if "location" in missing_slots:
        followups.append("Lokasi training (pabrik/kantor/hotel) dan kota?")
    if "preferred_dates" in missing_slots:
        followups.append("Tanggal target / bulan rencana?")

    header = f"Sepertinya terkait {topic_hint}." if topic_hint else "Saya temukan beberapa opsi:"
    lines = [header]
    if snippets:
        lines.append("🔎 Info relevan:")
        lines.extend(snippets)

    if followups:
        lines.append("Agar jawabannya pas, boleh lengkapi:")
        for q in followups[:3]:
            lines.append(f"• {q}")
    else:
        lines.append("Ada detail lain yang perlu saya bantu klarifikasi?")

    return "\n".join(lines)

def detect_intent(user_text: str):
    """Return (intent_name, debug_info) so thresholds can be tuned easily."""
    debug = {
        "clf_confidence": None,
        "semantic_best_intent": None,
        "semantic_similarity": None,
    }
    best_clf_intent, clf_conf = None, 0.0
    try:
        probs = intent_clf.predict_proba([user_text])[0]
        classes = intent_clf.named_steps["logreg"].classes_
        best_idx = int(np.argmax(probs))
        best_clf_intent = classes[best_idx]
        clf_conf = float(probs[best_idx])
        debug["clf_confidence"] = clf_conf
    except Exception:
        pass

    semantic_intent_label, semantic_score = semantic_intent(user_text)
    debug["semantic_best_intent"] = semantic_intent_label
    debug["semantic_similarity"] = semantic_score

    if best_clf_intent and clf_conf >= INTENT_CONFIDENCE_THRESHOLD:
        chosen = best_clf_intent
    elif semantic_intent_label and semantic_score >= SEMANTIC_INTENT_THRESHOLD:
        chosen = semantic_intent_label
    else:
        chosen = None

    debug["final_intent"] = chosen
    return chosen, debug

def respond(text, session_state=None):
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    state = ensure_session_state(session_state)
    intent, debug = detect_intent(text)
    slots = extract_slots(text)
    course_match = match_course_reference(text)

    intent = apply_context_rules(text, intent, debug, state, slots)
    update_state_from_slots(state, slots, course_match)
    clf_conf = debug.get("clf_confidence") or 0.0
    sem_conf = debug.get("semantic_similarity") or 0.0
    low_conf_router = (clf_conf < INTENT_CONFIDENCE_THRESHOLD) and (sem_conf < SEMANTIC_INTENT_THRESHOLD)

    if intent:
        state["current_intent"] = intent
        if intent == "external_training_request":
            state["mode"] = state.get("mode") or "external"

    fallback_needed = (intent is None) or (intent == "other") or low_conf_router

    if not fallback_needed and intent == "catalog":
        reply = handle_catalog()
    elif not fallback_needed and intent == "schedule":
        reply = handle_schedule(text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "pricing":
        reply = handle_pricing(text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "registration":
        reply = handle_registration(text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "external_training_request":
        reply = handle_external_training_request(text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "custom":
        reply = handle_custom(text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "policy":
        reply = handle_policy()
    elif not fallback_needed and intent == "contact":
        reply = handle_contact()
    else:
        rag_results = rag_search(text, k=5)
        detected_slots = slots.copy()
        if course_match:
            detected_slots["course_match"] = course_match
        missing_slots = detect_missing_slots(slots, state)
        reply = generate_fallback_response(text, rag_results, detected_slots, missing_slots)

    return reply, state

# =====================================================
# 🗂️ LEAD LOGGING
# =====================================================
LEADS_CSV = "leads.csv"
if not os.path.exists(LEADS_CSV):
    pd.DataFrame(columns=["ts","name","company","email","phone","interest","notes"]).to_csv(LEADS_CSV, index=False)

def save_lead(name, company, email, phone, interest, notes):
    df = pd.read_csv(LEADS_CSV)
    df.loc[len(df)] = [datetime.now().isoformat(), name, company, email, phone, interest, notes]
    df.to_csv(LEADS_CSV, index=False)

# =====================================================
# 🖥️ GRADIO UI
# =====================================================
# Quick suggestions
quick_msgs = [
    "Cara daftar training",
    "Lihat katalog pelatihan",
    "Jadwal terdekat JKK",
    "Harga TClass",
    "Daftar 10 pax untuk JKK-SV-101",
    "Bisa in-house di pabrik kami PT XYZ?",
    "Minta proposal in-house 20 orang di Karawang",
    "Kebijakan pembatalan",
    "Kontak & lokasi"
]
def chat_fn(history, message, session_state, name, company, email, phone, interest):
    msg = message or ""
    state = ensure_session_state(session_state)
    try:
        reply, updated_state = respond(msg, session_state=state)
    except Exception as e:
        reply = f"⚠️ Error: {e!s}"
        updated_state = state
    if interest and isinstance(msg, str) and any(k in msg.lower() for k in ["daftar","register","enroll","pesan"]):
        save_lead(name, company, email, phone, interest, msg)
    history = (history or []) + [(msg, reply)]
    return history, updated_state, ""

def main():
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown(f"### 🤖 {ORG_NAME} — Training Assistant\nHubungi: {CONTACT_EMAIL}")
        with gr.Row():
            with gr.Column(scale=3):
                name = gr.Textbox(label="Nama")
                company = gr.Textbox(label="Perusahaan")
                email = gr.Textbox(label="Email")
                phone = gr.Textbox(label="No HP")
                interest = gr.Dropdown(["","JKK","TClass","QCC","In-house"], label="Minat", value="")
            with gr.Column(scale=7):
                chat = gr.Chatbot(height=420)
                msg = gr.Textbox(label="Ketik pertanyaan…")
                send = gr.Button("Kirim", variant="primary")
        session_state = gr.State(init_session_state())
        send.click(chat_fn, [chat,msg,session_state,name,company,email,phone,interest], [chat,session_state,msg])
        msg.submit(chat_fn, [chat,msg,session_state,name,company,email,phone,interest], [chat,session_state,msg])
    demo.launch(share=True, debug=True, show_error=True)

if __name__ == "__main__":
    main()
