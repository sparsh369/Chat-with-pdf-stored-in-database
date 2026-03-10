"""
ingest.py  –  Run this ONCE (or whenever you add new PDFs).

Usage:
    python ingest.py                     # processes ./pdfs/ folder
    python ingest.py --pdf_dir my_docs   # custom folder

It reads every PDF, chunks + embeds the text, and persists the
vectors to ./chroma_db/  so app.py can use them without re-embedding.
"""

import os
import argparse
from pathlib import Path
import pypdf
import chromadb
from chromadb.config import Settings
from openai import OpenAI


# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "pdf_rag"


def split_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def ingest(pdf_dir: str):
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDFs found in '{pdf_dir}'. Put your PDFs there and re-run.")
        return

    # ── OpenAI client ──────────────────────────────────────────────────────────
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        raise ValueError("OPENAI_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
    client = OpenAI(api_key=api_key)

    # ── Persistent ChromaDB ────────────────────────────────────────────────────
    db = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection so re-running ingest is idempotent
    try:
        db.delete_collection(COLLECTION)
        print("[~] Deleted existing collection – rebuilding from scratch.")
    except Exception:
        pass
    collection = db.create_collection(COLLECTION)

    # ── Process each PDF ───────────────────────────────────────────────────────
    all_chunks: list[dict] = []

    for pdf_path in pdf_files:
        print(f"[+] Reading: {pdf_path.name}")
        reader = pypdf.PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                for chunk in split_text(text):
                    all_chunks.append({
                        "text"    : chunk,
                        "page"    : page_num + 1,
                        "source"  : pdf_path.name,
                    })

    print(f"[✓] Total chunks to embed: {len(all_chunks)}")

    # ── Embed in batches of 50 ─────────────────────────────────────────────────
    BATCH = 50
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
        print(f"    Embedded {min(i + BATCH, len(all_chunks))}/{len(all_chunks)} chunks…")

    print(f"\n✅ Done! Embeddings saved to '{CHROMA_DIR}/'")
    print(f"   PDFs ingested : {[f.name for f in pdf_files]}")
    print(f"   Total chunks  : {len(all_chunks)}")
    print("\nNow run:  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="./pdfs", help="Folder containing your PDFs")
    args = parser.parse_args()
    ingest(args.pdf_dir)
