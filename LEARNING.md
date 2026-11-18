# Learning guide: building an Agentic RAG assistant step by step

This file grows with the project. Each step explains **what** was built,
**why** it was designed that way, the **concepts** you need, and an
**exercise** so you can practise building it yourself.

| Step | Topic | Status |
|------|-------|--------|
| 1 | Ingestion: load, chunk, embed, store | ✅ done |
| 2 | Answering questions with citations | ⏳ next |
| 3 | The agent: routing, ranking, self-checking | ⏳ |
| 4 | Chat UI and Docker | ⏳ |
| 5 | Evaluation and a resume-ready README | ⏳ |

---

## The big picture

RAG stands for **Retrieval-Augmented Generation**. A plain LLM answers from
memory, which is frozen at training time and knows nothing about *your*
documents. It may also confidently invent facts ("hallucinate"). RAG fixes that
by *looking things up first*:

```
              INDEXING (once, ahead of time)                 QUERY (every question)
 files ─► load ─► chunk ─► embed ─► store in Qdrant    question ─► embed ─► search Qdrant
                                                                               │
                                                     answer + citations ◄─ LLM ◄┘
```

"Agentic" RAG adds a decision maker on top: an agent that chooses *where* to
look (your docs, the web, an API), checks whether the answer is really
supported, and tries again if not. We get there in step 3.

---

## Step 1: Ingestion

**Goal:** turn a folder of PDFs, Markdown, text and HTML files into small
pieces of text stored in a vector database, ready to be searched by meaning.

### What was built

| File | Job |
|------|-----|
| `src/rag_assistant/ingestion/loaders.py` | Reads each file type into plain text + metadata |
| `src/rag_assistant/ingestion/chunking.py` | Splits text into overlapping chunks |
| `src/rag_assistant/embeddings.py` | Turns text into vectors |
| `src/rag_assistant/vectorstore.py` | Stores and searches vectors in Qdrant |
| `src/rag_assistant/ingestion/pipeline.py` | Glues the steps together |
| `src/rag_assistant/cli.py` | `rag ingest`, `rag search`, `rag stats` commands |
| `src/rag_assistant/config.py` | All settings in one typed object |

Try it:

```bash
pip install -e ".[dev]"
rag ingest data/sample
rag search "how do vector databases compare vectors?"
```

### Concept 1: Loading, and why metadata matters

Every format needs its own reader: `pypdf` for PDFs, `BeautifulSoup` for
HTML (we throw away `<script>`, `<nav>` and so on, because that is noise),
plain reading for Markdown/text.

The important design decision: **a PDF becomes one `Document` per page**, and
every Document carries `source` and `metadata` (page, title). We do not need
that yet, but in step 2 the assistant must say *"according to handbook.pdf,
page 12"*. If you drop metadata at ingestion time you can never cite later.
Rule of thumb: **keep provenance from the very first step.**

### Concept 2: Chunking

An embedding model compresses a text into *one* vector. Compress a 50-page PDF
into one vector and it means "a bit of everything", so it matches nothing well.
Also, the LLM can only read a limited amount of text, and we want to send it
only the relevant paragraphs. So we cut documents into **chunks** of about
800 characters (roughly 150 to 200 words).

Our splitter is **recursive**:

1. Split on blank lines (paragraphs). Any piece still too long gets split on
   single newlines, then sentences (`". "`), then words, then a hard cut.
2. Greedily pack small pieces back together until the next one would exceed
   `chunk_size`.
3. Start every new chunk with the last `overlap` (120) characters of the
   previous chunk.

Why the overlap? Imagine the answer is a sentence that happens to sit across a
chunk boundary. Without overlap, half of it is in chunk 7 and half in chunk 8,
and neither matches well. Overlap makes sure each boundary sentence also appears
whole somewhere.

Trade-offs to remember for interviews:

* **Small chunks**: precise matches, but each one lacks context.
* **Large chunks**: more context, but blurrier vectors and more tokens sent to the LLM.
* 500 to 1000 characters with 10 to 20% overlap is a common starting point. Step 5
  measures whether it is a good choice for *our* data.

Libraries such as LangChain ship a `RecursiveCharacterTextSplitter` that does
the same thing. We wrote our own (about 50 lines) so you can see there is no
magic.

### Concept 3: Embeddings

An **embedding** is a list of numbers (here 384 of them) produced by a neural
network so that **texts with similar meaning get vectors pointing in similar
directions**. "How do I reset my password?" and "I forgot my login" share
almost no words, but their embeddings are close. That is why this is called
*semantic* search, compared to keyword search.

We compare vectors with **cosine similarity**: the cosine of the angle between
them. 1 means same direction (same meaning), 0 means unrelated.

We use `BAAI/bge-small-en-v1.5` through **fastembed**. It runs on your CPU, is
free, needs no API key, and ranks well for its size. One detail: bge models
expect questions and passages to be embedded slightly differently, which is why
`embed_query` and `embed_documents` are separate methods.

**Design pattern: the `Embedder` protocol.** The rest of the code only knows
"something with `embed_documents`, `embed_query` and `dim`". So we can have:

* `FastEmbedEmbedder`: the real model.
* `HashingEmbedder`: a toy that hashes words into buckets. It understands no
  meaning, but it needs no download, so the **tests run instantly and offline**.

Swapping in OpenAI or Voyage embeddings later is one new class. This is the
*dependency inversion* idea: depend on an interface, not a concrete library.

### Concept 4: The vector database (Qdrant)

Qdrant stores, per chunk, a **point**: an `id`, the `vector`, and a JSON
**payload** (text, source, page, title). A search embeds the question and asks
for the points with the highest cosine similarity.

Comparing against every vector is fine for thousands of chunks but too slow for
millions, so Qdrant builds an **HNSW index** (Hierarchical Navigable Small
World): a layered graph where a search hops greedily towards closer and closer
vectors. It is *approximate* nearest neighbour search: much faster, very rarely
misses the true best match.

Qdrant runs in two modes with the same Python API, which we exploit:

* **Embedded**: `QdrantClient(path="./qdrant_data")`. Just a folder, no server.
  This is the default, so the project works right after `pip install`.
* **Server**: `QdrantClient(url="http://localhost:6333")` with Docker. We switch
  to it in step 4.

### Concept 5: Idempotent ingestion

Running `rag ingest` twice must not store everything twice. Two tricks:

1. **Deterministic ids.** A chunk's id is a SHA-256 hash of
   `source + page + position + text`. Same chunk means same id, so the second
   upsert *overwrites* rather than duplicates. ("Upsert" = update or insert.)
2. **Replace per file.** Before storing a file's chunks we delete all its old
   chunks. Otherwise, if you shorten a file, chunks from the old ending stay
   behind and can surface in answers. `test_reingest_removes_stale_chunks`
   checks exactly this.

### Concept 6: Tests that do not need the internet

`tests/conftest.py` builds an **in-memory Qdrant** with the **hashing
embedder**, so all 15 tests run in a fraction of a second, with no Docker,
download or API key. That is what makes the GitHub Actions CI
(`.github/workflows/ci.yml`) cheap and reliable.

### Exercises

1. **Play with chunk size.** Run `RAG_CHUNK_SIZE=200 rag ingest data/sample`,
   then `rag stats`. Repeat with 2000. How does the chunk count change? Search
   the same question each time and compare the results.
2. **Feel the difference between real and fake embeddings.** Search
   `"comparing meaning of texts"` with `RAG_EMBEDDER=fastembed` and with
   `RAG_EMBEDDER=hashing`. Why does the hashing one struggle? (Hint: which
   words actually appear in the documents?) Delete `qdrant_data/` when you
   switch, because the two embedders produce incompatible vectors.
3. **Add a loader.** Add `.docx` support with the `python-docx` package: write
   `_load_docx`, register the suffix, and add a test like `test_html_strips_scripts`.
4. **Interview question to practise:** *"Why not embed the whole document?"*
   Explain it out loud using concepts 2 and 3.

### What's next

Step 2 adds the **G** in RAG: send the best chunks to an LLM with a prompt that
forces it to answer only from them and to cite them as `[1]`, `[2]`.
