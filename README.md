# 🎸 RAG’n’Roll

**A Local, Free, Daily Snapshot RAG System for Technology News**

---
## 🎥 Project Demo Video

▶️ **YouTube Link:**  
[https://www.youtube.com/watch?v=YOUR_VIDEO_ID](https://www.youtube.com/watch?v=6jSEGyZQi6w)

> This video demonstrates the full RAG’n’Roll pipeline, including:
> - News ingestion (RSS)
> - Embedding + FAISS indexing
> - Streamlit interface
> - Live query answering with citations

## 📌 Overview

**RAG’n’Roll** is a fully local **Retrieval-Augmented Generation (RAG)** system that answers questions about **recent technology news** using real articles ingested from RSS feeds.

Unlike cloud-based GenAI systems, RAG’n’Roll:

* runs **entirely locally**
* uses **no paid APIs**
* avoids hallucination by **grounding answers in retrieved sources**
* clearly shows **publication timestamps** to support “today” queries

This project was built for a **Practical Data Science / GenAI-focused course**, with an emphasis on real-world data pipelines, retrieval, and evaluation.

---

## 🚀 Key Features

* 📰 Live news ingestion via RSS feeds (Google News, NYT Tech, The Verge)
* 🧹 Text cleaning to remove site boilerplate and junk
* 🧠 Semantic retrieval using Sentence-Transformers + FAISS
* 🤖 Local LLM generation via Ollama (`llama3.2`)
* 📚 Source-grounded answers with citations
* 🕒 Visible publication dates for recency awareness
* 📊 Relevance@k metric (lightweight evaluation proxy)
* 🎨 Streamlit UI with refresh + logs

---

## 🧱 Architecture (Daily Snapshot RAG)

```
RSS Feeds (Google News / NYT / The Verge)
        |
        v
Ingestion (core/ingest.py) -> SQLite DB (articles)
        |
        v
Chunk + Embed (core/embed.py) -> chunks.jsonl + faiss.index
        |
        v
Retrieve (core/retrieve.py) -> top-k chunks
        |
        v
Generate (core/rag.py -> Ollama llama3.2)
        |
        v
Streamlit UI (app/app.py)
```

---

## 🛠️ Tech Stack

| Component     | Technology                           |
| ------------- | ------------------------------------ |
| Language      | Python 3.9+                          |
| UI            | Streamlit                            |
| Database      | SQLite                               |
| Embeddings    | sentence-transformers (MiniLM-L6-v2) |
| Vector Search | FAISS                                |
| LLM           | Ollama (llama3.2)                    |
| Parsing       | feedparser, trafilatura              |
| Evaluation    | Keyword-overlap Relevance@k          |

---

## ⚙️ Setup Instructions

### 1️⃣ Prerequisites

* Python 3.9+
* Ollama installed locally
  [https://ollama.com](https://ollama.com)

Pull the model:

```bash
ollama pull llama3.2
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the application

From the project root:

```bash
streamlit run app/app.py
```

---

## 🔄 Refreshing News (Daily Snapshot)

Use the **🔄 Refresh News** button in the UI.

This runs:

1. RSS ingestion (`core/ingest.py`)
2. Text cleaning
3. Chunking + embedding
4. FAISS index rebuild

This ensures the system answers based on the **latest available articles**.

---

## 🧪 Example Query

**Question:**
What is happening in technology news today?

**Answer:**

* Generated only from retrieved articles
* Includes inline citations `[1], [2], ...`
* Shows publication timestamps under each source

---

## 📊 Evaluation: Relevance@k

Since labeled relevance data is unavailable, the system uses a **simple proxy metric**:

**Relevance@k** = average keyword overlap between the query and retrieved snippets

This provides a rough indication of retrieval quality without requiring human labels.

---

## ⚠️ Known Limitations

* “Today” refers to the **latest ingested articles**, not real-time APIs
* RSS feeds vary in update frequency
* Some publisher boilerplate may still appear
* Relevance@k is **not a true IR metric**
* No cross-source deduplication or re-ranking

These tradeoffs are intentional to keep the system **local, transparent, and free**.

---

## 🔮 Future Improvements

* Time-based filtering (last 24h / 7 days)
* Cross-source deduplication
* LLM-based re-ranking
* Improved evaluation with human feedback
* Topic-based filtering (AI, hardware, policy)

---

## 👨‍💻 Author

**Krishna Kirit Maniyar**
MS in Data Science
Pace University

---

## 📄 License

MIT License
Free to use, modify, and extend with attribution.
