# Chat with PDF (RAG · GPT-4o · ChromaDB)

Chat with your local PDFs. Embeddings are pre-computed and persisted to disk.
Your API key never touches the browser — it stays in `.env`.

---

## Setup

```bash
# 1. Clone / download this project
git clone <your-repo>
cd chat-with-pdf

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Edit .env and paste your OpenAI key:  OPENAI_API_KEY=sk-...

# 5. Add your PDFs
mkdir pdfs
cp /path/to/your/files/*.pdf pdfs/

# 6. Ingest (run ONCE, or re-run when you add new PDFs)
python ingest.py

# 7. Launch the app
streamlit run app.py
```

---

## File structure

```
chat-with-pdf/
├── app.py            ← Streamlit UI
├── ingest.py         ← PDF → embeddings → ChromaDB
├── requirements.txt
├── .env              ← your secret key (git-ignored)
├── .env.example      ← template to commit
├── .gitignore
├── pdfs/             ← put your PDFs here (git-ignored)
└── chroma_db/        ← auto-created by ingest.py (git-ignored)
```

---

## Adding new PDFs

```bash
cp new_document.pdf pdfs/
python ingest.py      # rebuilds the whole DB
streamlit run app.py
```

---

## Notes

- `chroma_db/` and `pdfs/` are **git-ignored** — they stay on your machine only.
- The sidebar lets you filter which PDFs are searched per question.
- Re-running `ingest.py` wipes and rebuilds the collection (idempotent).
