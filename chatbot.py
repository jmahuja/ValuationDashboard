# app.py
import os
import json
import re
import fitz  # PyMuPDF
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any

# ----------------------------
# Configuration / Models
# ----------------------------
st.set_page_config(page_title="10-K RAG (local files)", layout="wide")

@st.cache_resource
def load_models():
    summarizer = pipeline(
        "text2text-generation",
        model="MBZUAI/LaMini-Flan-T5-248M",
        tokenizer="MBZUAI/LaMini-Flan-T5-248M"
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return summarizer, embedder

summarizer, embedder = load_models()

# ----------------------------
# Helpers: string conversions & cleaning
# ----------------------------
def ensure_string(x: Any) -> str:
    """
    Convert many common Python objects into a concise string safe for tokenizers/embedders.
    - dict -> "col1: val1 | col2: val2"
    - list/tuple -> "cell1 | cell2 | ..."
    - numbers, bool -> str(...)
    - None -> ""
    - if already str -> return as-is
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, dict):
        # join key:value pairs - skip empty values
        parts = []
        for k, v in x.items():
            if v is None or (isinstance(v, str) and v.strip() == ""):
                continue
            parts.append(f"{k}: {ensure_string(v)}")
        return " | ".join(parts)
    if isinstance(x, (list, tuple)):
        parts = [ensure_string(el) for el in x]
        parts = [p for p in parts if p]  # drop empties
        return " | ".join(parts)
    # fallback to JSON dump (compact)
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def clean_response(text: str) -> str:
    """Remove repeated lines and trim long repeated phrases."""
    if not text:
        return text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen = set()
    out = []
    for ln in lines:
        if ln not in seen:
            out.append(ln)
            seen.add(ln)
    return "\n".join(out)

# ----------------------------
# PDF extraction + chunking
# ----------------------------
def extract_pdf_text(pdf_path: str, max_pages: int = None) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        txt = page.get_text("text")
        # skip obvious metadata cover pages
        if "Commission File Number" in txt or "Exact name of registrant" in txt:
            continue
        texts.append(txt)
    return "\n".join(texts).strip()

def chunk_text_by_words(text: str, max_words: int = 220) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# ----------------------------
# JSON tables -> row strings
# ----------------------------
def prepare_json_tables(json_obj: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Input: parsed JSON object (list/dict)
    Output: list of tuples (row_str, metadata)
    metadata contains table_index, row_index optionally
    """
    rows_out = []

    def handle_table(table_obj, t_idx):
        # Several possible shapes: list of lists, list of dicts, dict with "rows"/"columns", etc.
        if isinstance(table_obj, list):
            # list could be rows; check first element
            if len(table_obj) == 0:
                return
            first = table_obj[0]
            if isinstance(first, dict):
                # list of dicts -> rows with keys
                for r_idx, row in enumerate(table_obj):
                    row_str = ensure_string(row)
                    meta = {"source": "table", "table_index": t_idx, "row_index": r_idx}
                    rows_out.append((row_str, meta))
            elif isinstance(first, (list, tuple)):
                # list of lists: try to create "col1:val1 | col2:val2" using header if possible
                # We cannot infer column names here; just join cells
                for r_idx, row in enumerate(table_obj):
                    row_str = ensure_string(row)
                    meta = {"source": "table", "table_index": t_idx, "row_index": r_idx}
                    rows_out.append((row_str, meta))
            else:
                # fallback: stringify each element
                for r_idx, row in enumerate(table_obj):
                    row_str = ensure_string(row)
                    meta = {"source": "table", "table_index": t_idx, "row_index": r_idx}
                    rows_out.append((row_str, meta))
        elif isinstance(table_obj, dict):
            # dict might have "columns"/"rows" or "data"
            if "rows" in table_obj and "columns" in table_obj:
                cols = table_obj.get("columns", [])
                for r_idx, row in enumerate(table_obj.get("rows", [])):
                    # associative if row length matches columns
                    if isinstance(row, (list, tuple)) and len(row) == len(cols):
                        pair = {cols[i]: row[i] for i in range(len(cols))}
                        row_str = ensure_string(pair)
                    else:
                        row_str = ensure_string(row)
                    meta = {"source": "table", "table_index": t_idx, "row_index": r_idx}
                    rows_out.append((row_str, meta))
            elif "data" in table_obj and isinstance(table_obj["data"], list):
                for r_idx, row in enumerate(table_obj["data"]):
                    row_str = ensure_string(row)
                    meta = {"source": "table", "table_index": t_idx, "row_index": r_idx}
                    rows_out.append((row_str, meta))
            else:
                # single dict -> one row
                row_str = ensure_string(table_obj)
                meta = {"source": "table", "table_index": t_idx, "row_index": 0}
                rows_out.append((row_str, meta))
        else:
            # unknown format: stringify
            row_str = ensure_string(table_obj)
            meta = {"source": "table", "table_index": t_idx, "row_index": 0}
            rows_out.append((row_str, meta))

    # If the top-level JSON is a list of tables
    if isinstance(json_obj, list):
        for t_idx, table in enumerate(json_obj):
            handle_table(table, t_idx)
    elif isinstance(json_obj, dict):
        # If dict has "tables" key, use it; else try to treat as single table
        if "tables" in json_obj and isinstance(json_obj["tables"], list):
            for t_idx, table in enumerate(json_obj["tables"]):
                handle_table(table, t_idx)
        else:
            handle_table(json_obj, 0)
    else:
        # fallback: one row
        rows_out.append((ensure_string(json_obj), {"source": "table", "table_index": 0, "row_index": 0}))

    return rows_out

# ----------------------------
# Build FAISS index (with metadata)
# ----------------------------
def build_faiss_index_with_meta(text_rows: List[Tuple[str, Dict[str,Any]]]) -> Tuple[faiss.IndexFlatL2, np.ndarray, List[Dict[str,Any]]]:
    """
    text_rows: list of (text, metadata)
    returns: index, embeddings_matrix, metadatas list
    """
    if not text_rows:
        return None, None, []
    corpus = [ensure_string(t) for t, _ in text_rows]
    metadatas = [m for _, m in text_rows]
    # embed in batches to avoid spikes
    embs = embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index, embs, metadatas

def semantic_search(index: faiss.IndexFlatL2, query: str, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    q = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q, k=min(k, index.ntotal))
    return D, I

def assemble_context_from_indices(I: np.ndarray, corpus: List[str], metadatas: List[Dict[str,Any]], max_chars: int = 3500) -> Tuple[str, List[Dict[str,Any]]]:
    """
    I: array returned by faiss search (shape (1,k))
    returns: concatenated context string and selected metadata
    """
    parts = []
    selected = []
    total = 0
    for idx in I[0]:
        if idx < 0 or idx >= len(corpus):
            continue
        snippet = corpus[idx]
        meta = metadatas[idx].copy() if metadatas else {}
        tag = f"[{meta.get('source','txt')}:{meta.get('table_index', meta.get('id','?'))}:{meta.get('row_index','')}]"
        block = f"{tag} {snippet}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        selected.append(meta)
        total += len(block)
    return "\n\n".join(parts), selected

# ----------------------------
# LLM answer prompt (more elaborate)
# ----------------------------
def llm_answer_from_context(question: str, context: str) -> str:
    prompt = (
        "You are a financial analysis assistant specializing in SEC 10-K filings. "
        "You have access to both:\n"
        "1. Narrative business and risk sections from the filing (PDF text).\n"
        "2. Structured tables with financial data (JSON rows).\n\n"

        "Your task:\n"
        "- Always search both sources.\n"
        "- If the answer involves a financial figure (revenue, expenses, assets, liabilities, cash flow, etc.), "
        "extract the exact number(s) from the JSON table context.\n"
        "- Preserve units and time periods (e.g., \"in millions\", \"FY 2023\").\n"
        "- If multiple numbers are present, compare them clearly.\n"
        "- If no numeric data is available, fall back to textual explanation.\n\n"

        "Answer format:\n"
        "- Start with a **direct numeric answer** if available.\n"
        "- Then provide a **short explanation** (1–2 sentences) in plain English.\n"
        "- Be concise and avoid repeating the same sentence.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )

    out = summarizer(prompt, max_new_tokens=350, do_sample=False, temperature=0.0)
    txt = out[0].get("generated_text") or out[0].get("summary_text") or ""
    return clean_response(txt)
# ----------------------------
# Streamlit UI
# ----------------------------
st.title("10-K RAG — Local PDF + JSON tables (row-level embeddings)")

col1, col2 = st.columns([2,1])
with col1:
    pdf_path = st.text_input("Local PDF path", value="sample_10k.pdf")
    json_path = st.text_input("Local tables JSON path", value="tables.json")
with col2:
    max_pages = st.slider("Pages to read (0 = all)", 0, 200, 10)
    chunk_words = st.slider("Chunk size (words)", 150, 600, 220)
    top_k = st.slider("Top-K for retrieval", 1, 12, 6)
    build_index = st.button("Build index")

# Build index
if build_index:
    if not os.path.exists(pdf_path):
        st.error("PDF path not found")
    elif not os.path.exists(json_path):
        st.error("JSON path not found")
    else:
        try:
            st.info("Extracting PDF text ...")
            pdf_text = extract_pdf_text(pdf_path, max_pages=None if max_pages==0 else max_pages)
            st.success(f"Extracted {len(pdf_text):,} characters")

            st.info("Chunking text ...")
            text_chunks = chunk_text_by_words(pdf_text, max_words=chunk_words)
            text_rows = []
            for i, ch in enumerate(text_chunks):
                text_rows.append((ch, {"source":"pdf", "id": f"pdf_{i}"}))
            st.success(f"Produced {len(text_chunks)} text chunks")

            st.info("Loading JSON tables and converting rows ...")
            with open(json_path, "r", encoding="utf-8") as f:
                json_obj = json.load(f)
            table_rows = prepare_json_tables(json_obj)
            st.success(f"Produced {len(table_rows)} table rows")

            st.info("Combining corpus and building FAISS index (this may take a moment)...")
            combined = text_rows + table_rows
            # ensure all items are strings in corpus (safety)
            corpus = [ensure_string(t) for t, _ in combined]
            metadatas = [m for _, m in combined]

            index, embs, metas = build_faiss_index_with_meta(list(zip(corpus, metadatas)))
            # store in session
            st.session_state["faiss_index"] = index
            st.session_state["corpus"] = corpus
            st.session_state["metadatas"] = metas
            st.success("FAISS index built and cached in session_state ✅")

        except Exception as e:
            st.exception(e)

# QA UI
st.markdown("---")

st.subheader("Ask the filing (semantic RAG)")

if "faiss_index" in st.session_state:
    query = st.text_input("Enter a question (e.g. 'What was unearned revenue in 2023?')")
    if query:
        with st.spinner("Retrieving top results..."):
            D, I = semantic_search(st.session_state["faiss_index"], query, k=top_k)
            context, used_meta = assemble_context_from_indices(I, st.session_state["corpus"], st.session_state["metadatas"], max_chars=3500)
            answer = llm_answer_from_context(query, context)
        st.markdown("**Answer:**")
        st.write(answer)

        with st.expander("Sources (retrieved)"):
            if used_meta:
                import pandas as pd
                rows = []
                for m, (dist_idx) in zip(used_meta, range(len(used_meta))):
                    rows.append({
                        "source": m.get("source"),
                        "table_index": m.get("table_index", ""),
                        "row_index": m.get("row_index", ""),
                    })
                st.table(pd.DataFrame(rows))
            else:
                st.write("No sources found.")
else:
    st.info("Build the index first (select paths and click Build index).")
