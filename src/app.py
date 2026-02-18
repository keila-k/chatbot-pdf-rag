from pathlib import Path
import json
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VS_DIR = Path("vectorstore")

def load_store():
    index_path = VS_DIR / "index.faiss"
    chunks_path = VS_DIR / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        st.error("Vectorstore não encontrado. Rode: python src/indexar_pdfs.py")
        st.stop()
    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return index, chunks

def search(query: str, index, chunks, model, top_k=5):
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        c = chunks[int(idx)]
        results.append({"score": float(score), "source": c["source"], "text": c["text"]})
    return results

st.set_page_config(page_title="Chatbot de PDFs (RAG)", layout="wide")
st.title("Chatbot baseado em PDFs (Embeddings + Busca Vetorial)")

st.markdown("1) Coloque PDFs em `pdfs/`  2) Rode `python src/indexar_pdfs.py`  3) Abra este app")

index, chunks = load_store()
model = SentenceTransformer(MODEL_NAME)

query = st.text_input("Pergunte algo sobre os PDFs:", placeholder="Ex.: Quais são os principais resultados do artigo X?")
top_k = st.slider("Quantidade de trechos recuperados (top_k)", 3, 10, 5)

if st.button("Buscar", type="primary") and query.strip():
    results = search(query, index, chunks, model, top_k=top_k)

    st.subheader("Trechos mais relevantes (busca vetorial)")
    for r in results:
        st.write(f"**Fonte:** {r['source']} | **Score:** {r['score']:.3f}")
        st.write(r["text"])
        st.divider()

    st.subheader("Resposta (baseada nos trechos)")
    st.write(
        "Com base nos trechos recuperados acima, a resposta deve estar relacionada aos pontos que se repetem "
        "ou que respondem diretamente à pergunta. Se você quiser, posso te ajudar a integrar um modelo de linguagem "
        "para sintetizar automaticamente uma resposta final citando as fontes."
    )