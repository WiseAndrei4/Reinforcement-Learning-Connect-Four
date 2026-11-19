# Hybrid Vector Search & Learned Reranking Engine

A two-stage hybrid information retrieval engine built from scratch—combining a custom **C++17 HNSW** approximate nearest-neighbor index, a **LaTeX-aware Okapi BM25** sparse retriever, **Reciprocal Rank Fusion (RRF)**, **PyTorch Bi-Encoder / Cross-Encoder** neural models, and a real-time **Recall@10 vs. QPS** telemetry desktop workbench.

No generative LLM wrappers or prompt-stuffing pipelines are used anywhere in the stack; every stage of ingestion, indexing, candidate retrieval, fusion, and neural reranking runs natively on local hardware.

---

## System Architecture

```text
                        +----------------------------------------------+
                        |           Document & Image Ingestion         |
                        |  * arXiv / Wikipedia JSONL & Markdown Chunks |
                        |  * Local ViT Math OCR (pix2tex LatexOCR)     |
                        +----------------------+-----------------------+
                                               |
                       +-----------------------+-----------------------+
                       v                                               v
      +---------------------------------+             +---------------------------------+
      |   Bi-Encoder (SciBERT, 768-d)   |             |    LaTeX-Aware BM25 Indexer     |
      |   Mean Pooling + L2 Normalize   |             |   Okapi BM25 (k1=1.5, b=0.75)   |
      +----------------+----------------+             +----------------+----------------+
                       v                                               v
      +---------------------------------+             +---------------------------------+
      |     C++17 HNSW Graph Index      |             |     Inverted Posting Lists      |
      |  nanobind (GIL-free) + Disk I/O |             |    Exact Lexical & Macro Match  |
      +----------------+----------------+             +----------------+----------------+
                       | Top-30 Dense                                  | Top-30 Sparse
                       +-----------------------+-----------------------+
                                               v
                        +----------------------------------------------+
                        |        Reciprocal Rank Fusion (RRF)          |
                        |     score(d) = sum_i 1 / (60 + rank_i(d))    |
                        +----------------------+-----------------------+
                                               | Top-25 Fused Candidates
                                               v
                        +----------------------------------------------+
                        |     Cross-Encoder Reranker (MiniLM-L-6)      |
                        |   Full Cross-Attention [CLS] Q [SEP] D [SEP] |
                        +----------------------+-----------------------+
                                               |
                                               v
                        +----------------------------------------------+
                        |   Ranked Results + Live Telemetry Dashboard  |
                        |   nDCG@10 * MRR@10 * Recall@10 vs. QPS       |
                        +----------------------------------------------+
```

---

## Core Engineering Components

* **C++17 HNSW Index & Zero-Copy Bindings (`cpp/hnsw.hpp`, `cpp/bindings.cpp`)**
  * Implements a multi-layer Hierarchical Navigable Small World graph from scratch with exponential level sampling $l = \lfloor -\ln(U(0,1)) \cdot m_L \rfloor$ where $m_L = 1 / \ln(M)$, defaulting to $M = 16$, $M_{\max 0} = 32$, and `ef_construction = 200`.
  * Stores vectors in a contiguous flat `std::vector<float>` buffer for CPU cache locality and computes cosine distance directly over L2-normalized embeddings.
  * Exposes `add_batch` and `search` to Python via `nanobind` zero-copy `ndarray` views while releasing the Python GIL (`nb::gil_scoped_release`) during graph traversal.
  * Supports custom binary serialization (`save` / `load` with `0x484E5357` header validation) for fast cold-start disk persistence.

* **Metric Learning & Neural Reranking (`ml/bi_encoder.py`, `ml/cross_encoder.py`)**
  * **Bi-Encoder:** Encodes queries and technical passages independently using `allenai/scibert_scivocab_uncased` with attention-masked mean pooling and L2 normalization, trained via temperature-scaled InfoNCE loss ($\tau = 0.05$) combining in-batch negatives and mined hard negatives:
    $$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\mathbf{q}_i^\top \mathbf{d}_i^+ / \tau)}{\sum_{j=1}^{B} \exp(\mathbf{q}_i^\top \mathbf{d}_j^+ / \tau) + \sum_{m=1}^{M} \exp(\mathbf{q}_i^\top \mathbf{d}_{i,m}^- / \tau)}$$
  * **Cross-Encoder Reranker:** Jointly encodes `[CLS] query [SEP] candidate [SEP]` pairs through `cross-encoder/ms-marco-MiniLM-L-6-v2` to score fine-grained token-level interactions across top fused candidates.

* **LaTeX-Aware BM25 & Reciprocal Rank Fusion (`ml/bm25.py`, `ml/hybrid.py`)**
  * Tokenizes both natural language terms and raw LaTeX commands (`\\[a-zA-Z]+|[a-zA-Z0-9_]+`) so symbolic queries (e.g., `\nabla`, `\mathcal{L}`, `\sum`) match verbatim alongside dense semantic embeddings.
  * Merges dense HNSW rankings and sparse BM25 rankings via Reciprocal Rank Fusion ($k = 60$) while preserving per-retriever rank provenance for result inspection.

* **IR Evaluation & Interactive Telemetry UI (`ml/metrics.py`, `UI_Driver.py`)**
  * Computes `nDCG@10`, `MRR@10`, and `Recall@10` against exact brute-force cosine similarity ground truth.
  * Built on `customtkinter` with asynchronous worker threads, interactive `efSearch` (`16` to `256`) beam-width tuning, stage toggles (`HNSW Dense`, `BM25 Sparse`, `Cross-Encoder Rerank`), local `pix2tex` (`LatexOCR`) image ingestion (`ocr_engine.py`), Markdown export (`exporter.py`), and an embedded `matplotlib` dashboard plotting **Recall@10 vs. QPS** curves and per-stage millisecond latency breakdowns.

---

## Repository Structure

```text
Study-Assistant/
├── cpp/
│   ├── hnsw.hpp                 # C++17 HNSW graph index & binary disk serialization
│   └── bindings.cpp             # nanobind Python C-extension bindings (GIL-free)
├── ml/
│   ├── bi_encoder.py            # PyTorch Bi-Encoder + InfoNCE hard-negative loss
│   ├── bm25.py                  # LaTeX-aware Okapi BM25 sparse retriever & persistence
│   ├── cross_encoder.py         # PyTorch Cross-Encoder top-K reranker
│   ├── hybrid.py                # Reciprocal Rank Fusion (RRF) with rank provenance
│   └── metrics.py               # nDCG@10 and MRR@10 evaluation metrics
├── storage/                     # Persisted index.hnsw, bm25.json, docstore.json, vectors.npy
├── CMakeLists.txt               # CMake build config (-O3 -march=native + .pyi stub generation)
├── pyproject.toml               # scikit-build-core packaging config
├── build_wiki_corpus.py         # Streaming Wikipedia STEM corpus chunker (JSONL)
├── ocr_engine.py                # Local ViT math formula OCR (pix2tex)
├── exporter.py                  # Markdown report & result exporter
└── UI_Driver.py                 # Desktop search inspector & Recall/QPS telemetry dashboard
```

---

## Build & Installation

### 1. System Prerequisites

* **Python:** 3.10+ (tested on Python 3.14)
* **C++ Compiler:** `g++` / `clang++` with C++17 support (or MSVC on Windows)
* **Build Tools:** `cmake`, `ninja`, `nanobind`, `scikit-build-core`

On Fedora / RHEL:
```bash
sudo dnf install -y gcc-c++ python3-devel
```

On Ubuntu / Debian:
```bash
sudo apt update && sudo apt install -y build-essential python3-dev
```

### 2. Install Dependencies & Compile the C++ HNSW Extension

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch transformers numpy matplotlib customtkinter pillow pix2tex datasets
pip install nanobind scikit-build-core ninja cmake
pip install --no-build-isolation .
```

The build compiles `cpp/bindings.cpp` with `-O3 -march=native` and installs both the native `hnsw_core` shared library and its generated `hnsw_core.pyi` type stub into `.venv`.

---

## Usage

### 1. Generate a Clean Technical Corpus

Stream and chunk STEM articles from Wikipedia into 180-word overlapping passages ready for indexing.
You can specify how many articles should be downloaded in the file. This is to show how the program works, to give some "lorem ipsum" type base-knowledge.
```bash
python build_wiki_corpus.py
```

This produces `wiki_tech_corpus.jsonl` containing `{"title": ..., "text": ...}` records.

### 2. Launch the Retrieval Engine & Telemetry Workbench

```bash
python UI_Driver.py
```

1. **Ingest Data:** Click **Ingest Corpus (JSONL/MD)** to load `wiki_tech_corpus.jsonl`, **Ingest Math Images (OCR)** to extract LaTeX from formula screenshots via `pix2tex`, or **Index Text from Editor** to insert custom notes. All vectors, HNSW graph layers, BM25 postings, and document metadata persist automatically to `storage/`.
2. **Search & Inspect Provenance:** Enter a technical query, toggle any combination of **HNSW Dense**, **BM25 Sparse**, and **Cross-Encoder Rerank**, and adjust the `efSearch` slider (`16` to `256`). Each result displays its Cross-Encoder logit, RRF score, HNSW rank, cosine similarity, and BM25 rank.
3. **Benchmark & Tune:** Open **Recall / QPS Dashboard** to sweep `efSearch` across `[16, 32, 64, 128, 256]`, inspect `Recall@10`, `nDCG@10`, `MRR@10`, `p50/p95` latency, and QPS, apply the optimal `efSearch` beam width in one click, or switch to the **Live Pipeline Latency Breakdown** tab to profile per-stage bottlenecks across recent queries.
