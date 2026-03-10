"""
app.py  –  Streamlit chat UI with auto-ingestion on startup.

PDFs are committed to the repo in the ./pdfs/ folder.
On first run, embeddings are built and saved to ./chroma_db/.
API key is read from Streamlit Cloud Secrets.

Run:
    streamlit run app.py
"""

import os
from pathlib import Path
import streamlit as st
import chromadb
import pypdf
from openai import OpenAI

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f0f0f; color: #e8e3db; }
[data-testid="stSidebar"] { background: #161616; border-right: 1px solid #2a2a2a; }

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem; color: #e8e3db;
    letter-spacing: -0.02em; margin-bottom: 0; line-height: 1.1;
}
.main-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem; color: #c8a96e;
    letter-spacing: 0.15em; text-transform: uppercase;
    margin-top: 4px; margin-bottom: 2rem;
}
.status-ready {
    background: #1a2e1a; border: 1px solid #2d5a2d; color: #6fcf6f;
    border-radius: 6px; padding: 8px 14px;
    font-family: 'DM Mono', monospace; font-size: 0.75rem; text-align: center;
}
.status-waiting {
    background: #2a1f0f; border: 1px solid #5a3d1a; color: #c8a96e;
    border-radius: 6px; padding: 8px 14px;
    font-family: 'DM Mono', monospace; font-size: 0.75rem; text-align: center;
}
.stButton > button {
    background: #c8a96e !important; color: #0f0f0f !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important; width: 100%;
}
hr { border-color: #2a2a2a !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0f0f0f; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "text-embedding-3-small"
LLM_MODEL     = "gpt-4o"
TOP_K         = 4
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "pdf_rag"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# ────────────────────────────────────────────────────────────────────────────
# ▼▼▼  ADD YOUR PDF FILENAMES HERE  ▼▼▼
# Place your PDF files in the ./pdfs/ folder in your repo
PDF_DIR   = "./pdfs"
PDF_FILES = [
    "document1.pdf",
    "document2.pdf",
    # "document3.pdf",   ← add more as needed
]
# ▲▲▲  THAT'S THE ONLY THING YOU NEED TO CHANGE WHEN ADDING NEW PDFs  ▲▲▲
# ────────────────────────────────────────────────────────────────────────────

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [
    ("chat_history", []),
    ("messages",     []),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ───────────────────────────────────────────────────────────────────
def split_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def get_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except KeyError:
        return None


def build_db(client) -> chromadb.Collection:
    """Read PDFs from PDF_FILES list, embed chunks, persist to CHROMA_DIR."""
    all_chunks: list[dict] = []

    for filename in PDF_FILES:
        pdf_path = Path(PDF_DIR) / filename
        if not pdf_path.exists():
            st.warning(f"⚠ PDF not found: {pdf_path} — skipping.")
            continue

        reader = pypdf.PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                for chunk in split_text(text):
                    all_chunks.append({
                        "text"  : chunk,
                        "page"  : page_num + 1,
                        "source": filename,
                    })

    if not all_chunks:
        st.error("No text could be extracted from the PDFs. Check your PDF_FILES list.")
        st.stop()

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = db.create_collection(COLLECTION)

    BATCH = 50
    progress = st.progress(0, text="Building embeddings…")
    for i in range(0, len(all_chunks), BATCH):
        batch      = all_chunks[i : i + BATCH]
        texts      = [c["text"] for c in batch]
        response   = client.embeddings.create(model=EMBED_MODEL, input=texts)
        embeddings = [r.embedding for r in response.data]
        collection.add(
            ids        = [str(i + j) for j in range(len(batch))],
            documents  = texts,
            embeddings = embeddings,
            metadatas  = [{"page": c["page"], "source": c["source"]} for c in batch],
        )
        progress.progress(
            min((i + BATCH) / len(all_chunks), 1.0),
            text=f"Embedding {min(i + BATCH, len(all_chunks))}/{len(all_chunks)} chunks…"
        )
    progress.empty()
    return collection


@st.cache_resource(show_spinner=False)
def load_or_build_db(_client) -> tuple:
    """
    Load existing ChromaDB if present, otherwise build from PDFs.
    @st.cache_resource ensures this runs only ONCE per app deployment.
    """
    db_exists = (
        os.path.exists(CHROMA_DIR) and
        any(Path(CHROMA_DIR).iterdir())
    )

    if db_exists:
        try:
            db         = chromadb.PersistentClient(path=CHROMA_DIR)
            collection = db.get_collection(COLLECTION)
            results    = collection.get(include=["metadatas"])
            sources    = sorted({m["source"] for m in results["metadatas"] if "source" in m})
            return collection, sources, "loaded"
        except Exception:
            pass  # DB corrupt — fall through to rebuild

    # Build fresh DB
    with st.spinner("📚 First run — building embeddings from PDFs… (this takes a minute)"):
        collection = build_db(_client)

    results = collection.get(include=["metadatas"])
    sources = sorted({m["source"] for m in results["metadatas"] if "source" in m})
    return collection, sources, "built"


# ── Ask ───────────────────────────────────────────────────────────────────────
def ask(question: str, collection, client, selected_sources: list[str]) -> tuple:
    q_emb = client.embeddings.create(
        model=EMBED_MODEL, input=[question]
    ).data[0].embedding

    where = {"source": {"$in": selected_sources}} if selected_sources else None

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        where=where,
        include=["documents", "metadatas"],
    )
    context     = "\n\n".join(results["documents"][0])
    source_info = [(m["source"], m["page"]) for m in results["metadatas"][0]]

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions based on the "
            "PDF document context below. Use ONLY this context to answer. "
            "If the answer is not in the context, say "
            "'I couldn\\'t find that in the document.'\n\n"
            f"Context:\n{context}"
        ),
    }
    messages = [system_msg] + st.session_state.chat_history + [
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, max_tokens=1024, temperature=0.2
    )
    answer = response.choices[0].message.content

    st.session_state.chat_history.append({"role": "user",      "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    return answer, source_info


# ── Bootstrap ─────────────────────────────────────────────────────────────────
client = get_openai_client()

if not client:
    st.error("⚠ OPENAI_API_KEY not found. Go to Streamlit Cloud → App Settings → Secrets and add it.")
    st.stop()

collection, pdf_sources, db_status = load_or_build_db(client)

if db_status == "built":
    st.toast(f"✅ Embeddings built from {len(pdf_sources)} PDF(s)!", icon="📄")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title">Chat<br>with PDF</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">RAG · GPT-4o · ChromaDB</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        f'<div class="status-ready">✦ Ready · {len(pdf_sources)} PDF(s)</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if pdf_sources:
        st.markdown("**Filter by PDF**")
        selected_sources = st.multiselect(
            "", options=pdf_sources, default=pdf_sources,
            label_visibility="collapsed",
        )
    else:
        selected_sources = []

    st.markdown("---")

    st.markdown("**Model**")
    LLM_MODEL = st.selectbox(
        "", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.messages     = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            parts = [f"{src} p.{pg}" for src, pg in msg["sources"]]
            st.caption("📎 " + "  ·  ".join(parts))

question = st.chat_input("Ask anything about your PDFs…")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, sources = ask(question, collection, client, selected_sources)
                st.markdown(answer)
                if sources:
                    parts = [f"{src} p.{pg}" for src, pg in sources]
                    st.caption("📎 " + "  ·  ".join(parts))

                st.session_state.messages.append({"role": "user",      "content": question})
                st.session_state.messages.append({"role": "assistant", "content": answer,
                                                   "sources": sources})
            except Exception as e:
                st.error(f"Error: {e}")
