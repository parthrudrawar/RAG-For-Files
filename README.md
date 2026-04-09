# RAG-For-Files

> Built to answer questions about any set of web pages without reading them manually — scrape URLs, embed the content, and query it conversationally using a local RAG pipeline.

---

## The Problem it Solves

Reading through multiple web pages to find specific information is slow. This project builds a RAG (Retrieval-Augmented Generation) pipeline on top of any list of URLs — scrape them, embed the content into a vector store, and query across all of them in a single chat interface. No manual reading required.

---

## How it Works

```
url.txt (list of URLs)
        ↓
scrapper.py  →  Scrapes each URL, extracts title + all links
        ↓
raw_data.json  →  Stores raw scraped output
        ↓
json_formatter.py  →  Fetches title of each linked page,
                       structures data for embedding
        ↓
formatted_data.json  →  Clean structured data ready to embed
        ↓
embedder.py  →  Embeds each section using all-MiniLM-L6-v2,
                builds FAISS index → saved to faiss_index/
        ↓
chatbot.py  →  On query: FAISS similarity search (top 3 docs)
               → context + history → Mixtral-8x7B prompt
               → answer + follow-up link suggestions
```

---

## Pipeline Breakdown

| File | Responsibility |
|------|---------------|
| `scrapper.py` | Reads `url.txt`, scrapes each URL with BeautifulSoup, extracts page title and all outbound links, saves to `raw_data.json` |
| `json_formatter.py` | For each scraped link, fetches the linked page's title, structures sections as `{title, url, text}`, saves to `formatted_data.json` |
| `embedder.py` | Loads `formatted_data.json`, creates `Document` objects per section, embeds with `all-MiniLM-L6-v2`, saves FAISS index locally |
| `chatbot.py` | Loads FAISS index, runs similarity search on user query (k=3), builds Mixtral prompt with context + last 2 conversation turns, returns answer + relevant links |

---

## Technical Decisions Worth Noting

**Separation of pipeline stages** — Scraping, formatting, embedding, and querying are intentionally split into four independent scripts. Each stage can be rerun without affecting the others — re-scrape without re-embedding, or swap the embedding model without re-scraping.

**FAISS for vector search** — Chosen for local, zero-cost similarity search. The index is persisted to disk (`faiss_index/`) so embeddings are not recomputed on every chatbot run.

**Embedding model** — `sentence-transformers/all-MiniLM-L6-v2` — lightweight, fast, runs on CPU. No GPU or API key required for the retrieval step.

**LLM with fallback** — Primary model is Mixtral-8x7B-Instruct with optional 4-bit quantization (`USE_4BIT=true`). If Mixtral fails to load, the chatbot automatically falls back to `distilgpt2` so the pipeline stays functional.

**Conversation memory** — The chatbot keeps the last 2 turns of history and includes them in every prompt, giving contextual continuity without overloading the context window.

**Graceful degradation** — If `url.txt` is missing or empty, the scraper falls back to sample URLs. If the FAISS index is missing, the chatbot returns a soft error instead of crashing.

---

## Project Structure

```
RAG-For-Files/
├── scrapper.py          # Stage 1 — scrape URLs → raw_data.json
├── json_formatter.py    # Stage 2 — format raw data → formatted_data.json
├── embedder.py          # Stage 3 — embed + build FAISS index
├── chatbot.py           # Stage 4 — query FAISS + generate answer
├── url.txt              # Input — one URL per line
├── raw_data.json        # Output of scrapper.py
├── formatted_data.json  # Output of json_formatter.py
├── requirement.txt
└── .env
```

---

## Environment Variables

```env
URLS_FILE=url.txt
FORMATTED_DATA_FILE=formatted_data.json
USE_4BIT=false        # Set to true to load Mixtral in 4-bit quantization (needs bitsandbytes)
```

---

## Setup & Running the Pipeline

```bash
git clone https://github.com/parthrudrawar/RAG-For-Files.git
cd RAG-For-Files

pip install -r requirement.txt

# Add your URLs — one per line
echo "https://example.com" >> url.txt

# Step 1 — Scrape
python scrapper.py

# Step 2 — Format
python json_formatter.py

# Step 3 — Embed
python embedder.py

# Step 4 — Chat
python chatbot.py
```

---

## url.txt Format

```
https://docs.python.org/3/
https://en.wikipedia.org/wiki/Retrieval-augmented_generation
https://huggingface.co/docs
```

One URL per line. Lines not starting with `http` are ignored automatically.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **BeautifulSoup** | HTML parsing and link extraction |
| **sentence-transformers/all-MiniLM-L6-v2** | Text embeddings (local, CPU) |
| **FAISS** | Vector similarity search |
| **Mixtral-8x7B-Instruct** | Answer generation (with distilgpt2 fallback) |
| **LangChain** | Document structure + FAISS integration |
| **python-dotenv** | Config via `.env` |

---

## Author

[parthrudrawar](https://github.com/parthrudrawar)
