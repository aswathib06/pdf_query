# app.py
import streamlit as st
import os
import io
import json
import tempfile
from typing import List
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import requests

# -----------------------
# Utilities for loading
# -----------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        raise RuntimeError("PyPDF2 not installed. Add it to requirements.") from e

    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t + "\n"
    return text

def extract_text_from_ipynb(file_bytes: bytes) -> str:
    data = json.load(io.BytesIO(file_bytes))
    text = ""
    for cell in data.get("cells", []):
        cell_type = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        if cell_type == "markdown":
            text += src + "\n\n"
        elif cell_type == "code":
            # include code as plain text (useful if Q/A in notebook)
            text += "# code:\n" + src + "\n\n"
    return text

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    b = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(b)
    elif name.endswith(".ipynb"):
        return extract_text_from_ipynb(b)
    else:
        # try treating as text
        try:
            return b.decode("utf-8")
        except:
            return ""

# -----------------------
# Chunking & embeddings
# -----------------------
def split_into_chunks(text: str, size: int = 800, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = text.replace("\r", " ")
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + size)
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

@st.cache_resource
def load_embedder(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def embed_texts(embedder, texts: List[str]) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros((0, embedder.get_sentence_embedding_dimension()))
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

# -----------------------
# Retriever
# -----------------------
def build_index(embeddings: np.ndarray):
    if embeddings.shape[0] == 0:
        return None
    nn = NearestNeighbors(n_neighbors=min(8, embeddings.shape[0]), metric="cosine")
    nn.fit(embeddings)
    return nn

def retrieve_top_k(query_emb: np.ndarray, nn: NearestNeighbors, k: int = 4):
    if nn is None:
        return []
    dists, idxs = nn.kneighbors(query_emb.reshape(1, -1), n_neighbors=min(k, nn._fit_X.shape[0]))
    return idxs.flatten().tolist(), dists.flatten().tolist()

# -----------------------
# LLM Calls (OpenAI & HuggingFace)
# -----------------------
def call_openai_chat(api_key: str, system_prompt: str, user_prompt: str, model="gpt-4o-mini") -> str:
    import openai
    openai.api_key = api_key
    # Use ChatCompletion (or adjust model)
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def call_hf_inference(api_key: str, model_id: str, prompt: str, max_tokens=512) -> str:
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.0}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    # HF returns either {'error':...} or a list
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(out["error"])
    if isinstance(out, list) and len(out) > 0:
        # Many HF models return [{'generated_text': '...'}]
        return out[0].get("generated_text", "").strip()
    return str(out)

# -----------------------
# Main QA function
# -----------------------
def answer_with_context(
    question: str,
    chunks: List[str],
    embedder,
    embeddings: np.ndarray,
    nn,
    provider: str,
    api_key: str,
    hf_model_id: str = "google/flan-t5-large",
    top_k: int = 4,
) -> str:
    # embed question
    q_emb = embedder.encode(question, convert_to_numpy=True)
    idxs, dists = retrieve_top_k(q_emb, nn, k=top_k)
    context = "\n\n---\n\n".join([f"Chunk {i}:\n{chunks[i]}" for i in idxs])
    system_prompt = (
        "You are an assistant that answers user questions using ONLY the provided context. "
        "If the answer is not contained in the context, say you don't know."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite the chunk numbers if relevant."
    if provider == "openai":
        return call_openai_chat(api_key, system_prompt, user_prompt)
    else:
        # provider == 'huggingface'
        prompt = system_prompt + "\n\n" + user_prompt
        return call_hf_inference(api_key, hf_model_id, prompt)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Gen-AI PDF Q&A — Streamlit (Provider: OpenAI/HuggingFace)")

st.title("Gen-AI Document Q&A ASK me Iam There FOR You ")
st.markdown(
    """
Upload a PDF or Jupyter notebook (.ipynb) and ask questions.  
Embeddings are computed locally using sentence-transformers and the LLM (OpenAI or HuggingFace) is used for generation.
"""
)

# Sidebar provider selection
provider = st.sidebar.selectbox("Select LLM Provider", ["openai", "huggingface"])
if provider == "openai":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model_choice = st.sidebar.text_input("OpenAI model (e.g., gpt-4o-mini)", value="gpt-4o-mini")
else:
    api_key = st.sidebar.text_input("HuggingFace API Key", type="password")
    model_choice = st.sidebar.text_input("HuggingFace model id (e.g., google/flan-t5-large, bigscience/bloom)", value="google/flan-t5-large")

# File input (accept local default path as example)
st.write("### Upload document (PDF / .ipynb)")
default_path = "/mnt/data/gen_ai_project_pdf (2).ipynb"  # your uploaded notebook path (example)
uploaded = st.file_uploader("Choose a PDF or .ipynb file", type=["pdf", "ipynb"], accept_multiple_files=False)

use_default = False
if uploaded is None:
    if os.path.exists(default_path):
        if st.button("Use example uploaded notebook (default)"):
            # Load default path bytes
            with open(default_path, "rb") as f:
                b = f.read()
            # Create a simple in-memory uploaded-like object
            uploaded = st.session_state.get("_default_file")
            if uploaded is None:
                # store bytes object in session_state to reuse
                uploaded = type("U", (), {})()
                uploaded.name = os.path.basename(default_path)
                uploaded.read = lambda: b
                st.session_state["_default_file"] = uploaded
            use_default = True
        else:
            st.info("Or upload a file above (you can also click 'Use example uploaded notebook (default)')")

if uploaded is not None:
    # Read and extract text
    with st.spinner("Extracting text..."):
        text = extract_text_from_file(uploaded)
    if not text:
        st.error("Could not extract text from the uploaded file.")
    else:
        st.success("Extracted text from document (length: %d characters)" % len(text))

        # Split into chunks & embeddings (cache heavy ops)
        size = st.sidebar.number_input("Chunk size (words)", min_value=200, max_value=2000, value=800, step=100)
        overlap = st.sidebar.number_input("Chunk overlap (words)", min_value=0, max_value=800, value=200, step=50)
        top_k = st.sidebar.slider("Retriever top-k chunks", 1, 8, 4)

        if "chunks" not in st.session_state or st.session_state.get("source_text") != text or st.session_state.get("size") != size or st.session_state.get("overlap") != overlap:
            with st.spinner("Chunking and embedding (this may take a moment)..."):
                chunks = split_into_chunks(text, size=size, overlap=overlap)
                embedder = load_embedder()
                embeddings = embed_texts(embedder, chunks)
                nn = build_index(embeddings)
                # cache
                st.session_state["chunks"] = chunks
                st.session_state["embeddings"] = embeddings
                st.session_state["nn"] = nn
                st.session_state["embedder_name"] = "all-MiniLM-L6-v2"
                st.session_state["source_text"] = text
                st.session_state["size"] = size
                st.session_state["overlap"] = overlap
        else:
            chunks = st.session_state["chunks"]
            embeddings = st.session_state["embeddings"]
            nn = st.session_state["nn"]
            embedder = load_embedder()

        st.write(f"Document split into **{len(chunks)}** chunks.")

        # show a sample chunk
        if st.checkbox("Show first 2 chunks"):
            for i, c in enumerate(chunks[:2]):
                st.markdown(f"**Chunk {i}**\n\n```\n{c[:1000]}\n```")

        # Query box
        question = st.text_input("Ask a question about the document:")
        if st.button("Get Answer") and question.strip():
            if not api_key:
                st.error("Please provide the API key for the provider you selected in the sidebar.")
            else:
                with st.spinner("Retrieving context and querying model..."):
                    response = answer_with_context(
                        question=question,
                        chunks=chunks,
                        embedder=embedder,
                        embeddings=embeddings,
                        nn=nn,
                        provider=provider,
                        api_key=api_key,
                        hf_model_id=model_choice,
                        top_k=top_k,
                    )
                st.write("### Answer")
                st.write(response)

                st.write("---")
                st.write("### Retrieved chunks (for transparency)")
                q_emb = embedder.encode(question, convert_to_numpy=True)
                idxs, dists = retrieve_top_k(q_emb, nn, k=top_k)
                for i, d in zip(idxs, dists):
                    st.markdown(f"**Chunk {i} (dist={d:.4f})**")
                    st.write(chunks[i][:800] + ("..." if len(chunks[i]) > 800 else ""))

else:
    st.info("Upload a file to get started.")

