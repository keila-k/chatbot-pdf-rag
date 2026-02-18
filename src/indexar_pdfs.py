from pathlib import Path
import re
import json
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

PDF_DIR = Path("pdfs")
OUT_DIR = Path("vectorstore")
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# LIMITES DE SEGURANÇA (ajuste se quiser)
MAX_TEXT_PER_PDF_CHARS = 250_000     # evita PDF gigante explodir RAM
MAX_CHUNKS_PER_PDF = 400             # evita indexar demais em 1 PDF
EMBED_BATCH_SIZE = 16                # lote menor = menos RAM

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    total = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        text = clean_text(text)
        if not text:
            continue
        total += len(text)
        if total > MAX_TEXT_PER_PDF_CHARS:
            break
        parts.append(text)
    return " ".join(parts)

def chunk_text(text: str, source: str) -> list[dict]:
    chunks = []
    start = 0
    n = len(text)
    while start < n and len(chunks) < MAX_CHUNKS_PER_PDF:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"source": source, "text": chunk})
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
        if start >= n:
            break
    return chunks

def main():
    if not PDF_DIR.exists():
        raise SystemExit("Pasta 'pdfs/' não existe.")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("Nenhum PDF encontrado em 'pdfs/'.")

    model = SentenceTransformer(MODEL_NAME)

    # vamos inicializar o índice FAISS depois que soubermos a dimensão
    index = None
    all_chunks_meta = []

    pdf_count_ok = 0

    for pdf_path in pdfs:
        text = read_pdf_text(pdf_path)
        if not text:
            print(f"[SKIP] Sem texto extraível: {pdf_path.name}")
            continue

        chunks = chunk_text(text, pdf_path.name)
        if not chunks:
            print(f"[SKIP] Sem chunks gerados: {pdf_path.name}")
            continue

        pdf_count_ok += 1
        print(f"[PDF] {pdf_path.name} -> {len(chunks)} chunks")

        # embeddings em lotes pequenos e adiciona no FAISS incrementalmente
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i:i + EMBED_BATCH_SIZE]
            batch_texts = [c["text"] for c in batch]

            emb = model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            emb = np.asarray(emb, dtype="float32")

            if index is None:
                dim = emb.shape[1]
                index = faiss.IndexFlatIP(dim)

            index.add(emb)
            all_chunks_meta.extend(batch)

    if index is None or index.ntotal == 0:
        raise SystemExit("Nenhum conteúdo indexado. PDFs podem ser imagens/scans sem texto.")

    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    (OUT_DIR / "chunks.json").write_text(
        json.dumps(all_chunks_meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Indexação concluída.")
    print(f"- PDFs processados: {pdf_count_ok}/{len(pdfs)}")
    print(f"- Total chunks indexados: {index.ntotal}")
    print(f"- Saída: {OUT_DIR}/index.faiss e {OUT_DIR}/chunks.json")

if __name__ == "__main__":
    main()