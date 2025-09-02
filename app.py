import os
import json
import re
import fitz  # PyMuPDF
import pdfplumber
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------- Load Models ----------------------
@st.cache_resource
def load_summarizer():
    return pipeline(
        "text2text-generation",
        model="MBZUAI/LaMini-Flan-T5-248M",
        tokenizer="MBZUAI/LaMini-Flan-T5-248M"
    )

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

summarizer = load_summarizer()
embedder = load_embedder()

# ---------------------- PDF Utilities ----------------------
def extract_text(uploaded_file, max_pages=10):
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        text += page.get_text("text") + "\n"
    return text.strip()

def extract_tables(uploaded_file, max_pages=10):
    uploaded_file.seek(0)
    tables = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            if max_pages and i >= max_pages:
                break
            for table in page.extract_tables():
                if table:
                    tables.append(table)
    return tables

# ---------------------- Chunking ----------------------
def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# ---------------------- Pros & Cons ----------------------
def generate_pros_cons_from_chunks(chunks):
    pros, cons = [], []

    for chunk in chunks:
        prompt = (
            "Classify the following company disclosure text into strengths/opportunities "
            "and risks/weaknesses. Provide concise bullet points. "
            "Do not repeat statements. If nothing relevant, return 'None'.\n\n"
            f"{chunk}"
        )
        result = summarizer(prompt, max_length=300, min_length=100, do_sample=False)
        output = result[0].get("summary_text") or result[0].get("generated_text", "")

        if "opportunity" in output.lower() or "strength" in output.lower():
            pros.append(output)
        if "risk" in output.lower() or "weakness" in output.lower():
            cons.append(output)

    return "\n".join(pros), "\n".join(cons)

# ---------------------- Embeddings & Search ----------------------
def build_vector_store(chunks, tables):
    data = []

    # text chunks
    for i, ch in enumerate(chunks):
        data.append({"id": f"text_{i}", "content": ch, "type": "text"})

    # tables (flatten rows)
    for ti, table in enumerate(tables):
        for ri, row in enumerate(table):
            row_str = " | ".join(str(c) for c in row if c)
            data.append({"id": f"table_{ti}_{ri}", "content": row_str, "type": "table"})

    texts = [d["content"] for d in data]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return data, embeddings

def semantic_search(query, data, embeddings, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = np.dot(embeddings, q_emb)
    top_idx = sims.argsort()[-top_k:][::-1]
    return [data[i] for i in top_idx]

# ---------------------- Streamlit App ----------------------
st.title("📊 10-K Filing Analyzer")

pdf_file = st.file_uploader("Upload 10-K PDF", type=["pdf"])
json_file = st.file_uploader("Upload JSON Table Data", type=["json"])

if pdf_file:
    max_pages = st.slider("Limit pages (for testing)", 1, 50, 5)

    with st.spinner("Extracting text..."):
        text = extract_text(pdf_file, max_pages=max_pages)

    with st.spinner("Extracting tables..."):
        tables = extract_tables(pdf_file, max_pages=max_pages)

    if json_file:
        with st.spinner("Loading JSON tables..."):
            json_tables = json.load(json_file)
            # assume JSON is list of rows
            tables.extend(json_tables)

    with st.spinner("Splitting into chunks..."):
        chunks = chunk_text(text)

    with st.spinner("Building vector store..."):
        data, embeddings = build_vector_store(chunks, tables)


    # Pros & Cons
    if "pros" not in st.session_state or "cons" not in st.session_state:
        with st.spinner("Extracting positives & negatives..."):
            st.session_state.pros, st.session_state.cons = generate_pros_cons_from_chunks(chunks)

    st.subheader("✅ Positives")
    st.markdown(st.session_state.pros if st.session_state.pros else "[No positives extracted]")

    st.subheader("❌ Negatives")
    st.markdown(st.session_state.cons if st.session_state.cons else "[No negatives extracted]")

