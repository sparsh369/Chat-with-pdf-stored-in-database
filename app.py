"""
app.py  –  Streamlit chat UI.

Loads pre-built ChromaDB embeddings (created by ingest.py).
API key is read from .env – never typed in the browser.

Run:
    streamlit run app.py
"""

import os
import streamlit as st
import chromadb
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
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL   = "gpt-4o"
TOP_K       = 4
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "pdf_rag"

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [
    ("chat_history", []),
    ("messages",     []),
    ("collection",   None),
    ("db_ready",     False),
    ("pdf_sources",  []),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Load ChromaDB once ────────────────────────────────────────────────────────
@st.cache_resource
def load_db():
    """Load the persistent ChromaDB collection built by ingest.py."""
    if not os.path.exists(CHROMA_DIR):
        return None, []
    try:
        db         = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = db.get_collection(COLLECTION)
        # Collect unique source PDF names stored in metadata
        results  = collection.get(include=["metadatas"])
        sources  = sorted({m["source"] for m in results["metadatas"] if "source" in m})
        return collection, sources
    except Exception:
        return None, []

@st.cache_resource
    def get_openai_client():
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            return OpenAI(api_key=api_key)
        except KeyError:
            return None

collection, pdf_sources = load_db()
client                  = get_openai_client()
db_ready                = collection is not None and client is not None

# ── Ask function ──────────────────────────────────────────────────────────────
def ask(question: str, selected_sources: list[str]) -> tuple[str, list]:
    # Embed the question
    q_emb = client.embeddings.create(
        model=EMBED_MODEL, input=[question]
    ).data[0].embedding

    # Optional: filter by selected PDF sources
    where = {"source": {"$in": selected_sources}} if selected_sources else None

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        where=where,
        include=["documents", "metadatas"],
    )
    context      = "\n\n".join(results["documents"][0])
    source_info  = [(m["source"], m["page"]) for m in results["metadatas"][0]]

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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title">Chat<br>with PDF</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">RAG · GPT-4o · ChromaDB</p>', unsafe_allow_html=True)
    st.markdown("---")

    # DB status
    if not os.path.exists(CHROMA_DIR):
        st.markdown(
            '<div class="status-waiting">⚠ No DB found.<br>Run ingest.py first.</div>',
            unsafe_allow_html=True,
        )
    elif not client:
        st.markdown(
            '<div class="status-waiting">⚠ OPENAI_API_KEY missing.<br>Check your .env</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-ready">✦ DB ready · {len(pdf_sources)} PDF(s)</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Source filter
    if pdf_sources:
        st.markdown("**Filter by PDF**")
        selected_sources = st.multiselect(
            "", options=pdf_sources, default=pdf_sources,
            label_visibility="collapsed",
        )
    else:
        selected_sources = []

    st.markdown("---")

    # Model selector
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


# ── Main area ─────────────────────────────────────────────────────────────────
if not db_ready:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:60vh;text-align:center;opacity:0.5;">
        <div style="font-size:4rem;margin-bottom:1rem;">📄</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#e8e3db;">
            Run <code>python ingest.py</code> first
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.75rem;
                    color:#c8a96e;margin-top:8px;letter-spacing:0.1em;">
            PUT PDFs IN ./pdfs/ → RUN INGEST → CHAT
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                parts = [f"{src} p.{pg}" for src, pg in msg["sources"]]
                st.caption("📎 " + "  ·  ".join(parts))

    # Chat input
    question = st.chat_input("Ask anything about your PDFs…")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer, sources = ask(question, selected_sources)
                    st.markdown(answer)
                    if sources:
                        parts = [f"{src} p.{pg}" for src, pg in sources]
                        st.caption("📎 " + "  ·  ".join(parts))

                    st.session_state.messages.append({"role": "user",      "content": question})
                    st.session_state.messages.append({"role": "assistant", "content": answer,
                                                       "sources": sources})
                except Exception as e:
                    st.error(f"Error: {e}")
