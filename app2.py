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
        "label": "policy",
        "examples": ["batal", "refund", "reschedule", "pembatalan"],
        "description": "Menanyakan kebijakan pembatalan atau reschedule"
    },
    {
        "label": "contact",
        "examples": ["hubungi", "whatsapp", "nomor", "email"],
        "description": "Meminta kontak person atau channel komunikasi"
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
COURSE_RX = re.compile(r"(JKK|TCLASS|QCC)", re.I)
COMPANY_RX = re.compile(r"(PT\s+[A-Za-z0-9.&()\-\s]+)", re.I)

def extract_slots(text):
    s = {}
    if m:=PAX_RX.search(text): s["pax"]=int(m.group(1))
    if m:=COURSE_RX.search(text): s["course"]=m.group(1).upper()
    if m:=COMPANY_RX.search(text): s["company"]=m.group(1).strip()
    return s

# =====================================================
# 💬 HANDLERS
# =====================================================
def handle_catalog():
    out = "\n".join(f"- **{r.code} – {r.title}** ({r.format}, {r.duration_days} hari) @ {r.location} • Rp{int(r.price_idr):,}"
                    for _, r in courses.iterrows())
    return "📘 Program yang tersedia:\n" + out

def handle_schedule(text):
    slots = extract_slots(text)
    course = slots.get("course")
    if course:
        match = courses[courses.code.str.contains(course, case=False)]
    else:
        match = courses
    msg = ["📅 Jadwal:"]
    for _, r in match.iterrows():
        msg.append(f"- {r.code}: {r.next_runs} @ {r.location}")
    return "\n".join(msg)

def handle_pricing(text):
    slots = extract_slots(text)
    c = slots.get("course")
    if not c: return "Sebutkan kode pelatihan (contoh: JKK) untuk info harga."
    match = courses[courses.code.str.contains(c, case=False)]
    if match.empty: return "Kode tidak ditemukan."
    r = match.iloc[0]
    return f"Harga {r.code} – {r.title}: Rp{int(r.price_idr):,} / peserta."

def handle_registration(text):
    slots = extract_slots(text)
    comp   = slots.get("company")
    course = slots.get("course")
    pax    = slots.get("pax")
    # Build ringkas untuk header
    hdr = []
    if comp:   hdr.append(f"• Perusahaan: {comp}")
    if course: hdr.append(f"• Kursus: {course}")
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
f"> Batch: 2025-12-03\n"
f"> Kontak: 08xx-xxxx\n"
    )

def handle_custom(_): 
    return f"In-house bisa! Kirim kebutuhan ke {CONTACT_EMAIL} atau WA {WHATSAPP_LINK}"

def handle_policy(): 
    return faqs.iloc[3].a

def handle_contact(): 
    return f"Hubungi kami di {CONTACT_EMAIL} atau WA {WHATSAPP_LINK}."

def handle_rag(text):
    hits = rag_search(text, k=4)
    lines = ["🔎 Saya temukan info terkait:"]
    for score, _, meta_info in hits:
        if meta_info["type"] == "course":
            r = courses[courses.code == meta_info["code"]].iloc[0]
            lines.append(
                f"• {r.code} – {r.title}: {r.format}, {r.duration_days} hari. "
                f"Jadwal {r.next_runs} @ {r.location}. Harga Rp{int(r.price_idr):,}."
            )
        else:
            faq_row = faqs.iloc[meta_info["id"]]
            lines.append(f"• {faq_row.q} → {faq_row.a}")
    lines.append("Jika masih kurang sesuai, boleh jelaskan sedikit detail (kode kursus, jumlah peserta, atau tujuan training).")
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

def respond(text):
    intent, _debug = detect_intent(text)

    if intent == "catalog": return handle_catalog()
    if intent == "schedule": return handle_schedule(text)
    if intent == "pricing": return handle_pricing(text)
    if intent == "registration": return handle_registration(text)
    if intent == "custom": return handle_custom(text)
    if intent == "policy": return handle_policy()
    if intent == "contact": return handle_contact()
    # fallback to retrieval
    return handle_rag(text)

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
    "Kebijakan pembatalan",
    "Kontak & lokasi"
]
def chat_fn(history, message, name, company, email, phone, interest):
    msg = message or ""
    try:
        reply = respond(msg)
    except Exception as e:
        reply = f"⚠️ Error: {e!s}"
    if interest and any(k in msg.lower() for k in ["daftar","register","enroll","pesan"]):
        save_lead(name, company, email, phone, interest, msg)
    history = (history or []) + [(msg, reply)]
    return history, ""

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
        state = gr.State([])
        send.click(chat_fn, [state,msg,name,company,email,phone,interest], [chat,msg])
        msg.submit(chat_fn, [state,msg,name,company,email,phone,interest], [chat,msg])
    demo.launch(share=True, debug=True, show_error=True)

if __name__ == "__main__":
    main()
