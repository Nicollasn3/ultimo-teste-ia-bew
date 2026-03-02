"""
03_build_embeddings.py
-----------------------
Lê o OCR markdown (output/ocr_output.md), divide em chunks com metadados
(página, autor, linhas), gera embeddings com Qwen3-Embedding-4B e salva
um índice FAISS.

Saída:
  output/chunks/chunks.json      — lista de chunks com metadados completos
  output/chunks/faiss.index      — índice vetorial FAISS (float32, cosine)
  output/chunks/embedding_meta.json

Uso:
    python pipeline/03_build_embeddings.py [--md output/ocr_output.md]
                                           [--out output/chunks]
                                           [--model Qwen/Qwen3-Embedding-4B]
                                           [--chunk-size 400]
                                           [--overlap 80]
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    sys.exit("[ERRO] transformers/torch/numpy não instalados.")

try:
    import faiss
except ImportError:
    sys.exit("[ERRO] faiss não instalado. Execute: pip install faiss-gpu  (ou faiss-cpu)")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================================
# 1. Parsing do markdown de OCR
# ============================================================================

_PAGE_RE = re.compile(
    r"^## Página (\d+)\s*\n"
    r"<!-- source: page=\d+ author=\"([^\"]+)\" title=\"([^\"]+)\" -->\s*\n",
    re.MULTILINE,
)


def parse_markdown(md_text: str) -> List[Dict[str, Any]]:
    """
    Extrai blocos de texto por página do markdown gerado pelo step 02.
    Retorna lista de dicts: {page, author, title, text}
    """
    parts = _PAGE_RE.split(md_text)
    # parts = [pre, page_num, author, title, content, page_num, author, title, content, ...]
    pages = []
    i = 1
    while i + 3 < len(parts):
        page_num = int(parts[i])
        author   = parts[i + 1].strip()
        title    = parts[i + 2].strip()
        content  = parts[i + 3].strip()
        # Remove separadores de página do conteúdo
        content  = content.replace("---", "").strip()
        pages.append({
            "page":   page_num,
            "author": author,
            "title":  title,
            "text":   content,
        })
        i += 4

    if not pages:
        # Fallback: tenta extrair por separadores "---"
        raw_pages = [p.strip() for p in md_text.split("---") if p.strip()]
        for idx, raw in enumerate(raw_pages[1:], start=1):  # pula header
            pages.append({
                "page":   idx,
                "author": "Ricardo Marcelo Fonseca",
                "title":  "Introdução Teórica à História do Direito",
                "text":   raw,
            })

    return pages


# ============================================================================
# 2. Chunking com rastreamento de linhas
# ============================================================================

def chunk_page(
    page_data: Dict[str, Any],
    chunk_size: int = 400,     # tokens aproximados (palavras * 1.3)
    overlap_size: int = 80,
) -> List[Dict[str, Any]]:
    """
    Divide o texto de uma página em chunks sobrepostos.
    Rastreia linhas iniciais e finais de cada chunk.
    """
    text = page_data["text"]
    lines = text.splitlines()

    # Tokeniza de forma simples por palavras (evita dependência de tokenizer aqui)
    words_per_line: List[List[str]] = [ln.split() for ln in lines]

    # Achata em (palavra, linha_global)
    flat: List[tuple[str, int]] = []
    for line_idx, words in enumerate(words_per_line):
        for word in words:
            flat.append((word, line_idx + 1))  # linhas 1-indexed

    if not flat:
        return []

    words_only = [w for w, _ in flat]
    line_nums  = [l for _, l in flat]

    chunks = []
    chunk_id = 0
    i = 0

    while i < len(words_only):
        j = min(i + chunk_size, len(words_only))
        chunk_words = words_only[i:j]
        chunk_lines = line_nums[i:j]
        chunk_text  = " ".join(chunk_words)

        chunks.append({
            "chunk_id":   f"p{page_data['page']:04d}_c{chunk_id:03d}",
            "page":       page_data["page"],
            "author":     page_data["author"],
            "title":      page_data["title"],
            "start_line": chunk_lines[0],
            "end_line":   chunk_lines[-1],
            "text":       chunk_text,
        })

        chunk_id += 1
        i += chunk_size - overlap_size  # avança com sobreposição

    return chunks


def build_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for page in pages:
        page_chunks = chunk_page(page, chunk_size=chunk_size, overlap_size=overlap)
        all_chunks.extend(page_chunks)
    print(f"[03] Chunks criados: {len(all_chunks)} "
          f"(~{len(all_chunks) / len(pages):.1f} por página)")
    return all_chunks


# ============================================================================
# 3. Embeddings com Qwen3-Embedding-4B
# ============================================================================
# Qwen3 embedding usa last-token pooling com prompt de instrução.
# Ref: https://huggingface.co/Qwen/Qwen3-Embedding-4B

EMBED_INSTRUCTION = "Represent this passage for retrieval:"


def _last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool no último token não-padding."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden.shape[0]
    return last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]


def encode_texts(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    instruction: str = "",
) -> np.ndarray:
    """Gera embeddings normalizados para uma lista de textos."""
    if instruction:
        texts = [f"{instruction}\n{t}" for t in texts]

    all_embeddings = []
    device = next(model.parameters()).device
    iterator = range(0, len(texts), batch_size)
    if tqdm:
        iterator = tqdm(iterator, desc="  Embedding", unit="batch", leave=False)

    for start in iterator:
        batch = texts[start: start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(**encoded)

        embeddings = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])

        # L2 normalização → cosine similarity = inner product
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        all_embeddings.append(embeddings.cpu().float().numpy())

    return np.vstack(all_embeddings)


# ============================================================================
# 4. Índice FAISS
# ============================================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine (embeddings normalizados)
    faiss.normalize_L2(embeddings)   # garante L2=1
    index.add(embeddings)
    return index


# ============================================================================
# 5. Pipeline principal
# ============================================================================

def build_embeddings(
    md_path: Path,
    out_dir: Path,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    chunk_size: int = 400,
    overlap: int = 80,
    batch_size: int = 8,
) -> None:
    if not md_path.exists():
        sys.exit(f"[ERRO] Markdown não encontrado: {md_path}. Execute o step 02.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Parsing + chunking
    # ------------------------------------------------------------------
    print(f"\n[03] Lendo: {md_path}")
    md_text = md_path.read_text(encoding="utf-8")
    pages   = parse_markdown(md_text)
    print(f"[03] Páginas extraídas: {len(pages)}")

    chunks = build_chunks(pages, chunk_size=chunk_size, overlap=overlap)
    chunks_path = out_dir / "chunks.json"
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[03] Chunks salvos em: {chunks_path}")

    # ------------------------------------------------------------------
    # Carrega modelo de embeddings
    # ------------------------------------------------------------------
    print(f"\n[03] Carregando tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"[03] Carregando modelo de embeddings...")
    embed_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    embed_model.eval()
    print(f"[03] Modelo carregado em: {next(embed_model.parameters()).device}")

    # ------------------------------------------------------------------
    # Gera embeddings dos chunks
    # ------------------------------------------------------------------
    print(f"\n[03] Gerando embeddings para {len(chunks)} chunks...")
    t0 = time.time()
    texts = [c["text"] for c in chunks]
    embeddings = encode_texts(
        texts,
        model=embed_model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        instruction=EMBED_INSTRUCTION,
    )
    elapsed = time.time() - t0
    print(f"[03] Embeddings gerados: shape={embeddings.shape} | {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Constrói e salva índice FAISS
    # ------------------------------------------------------------------
    print("[03] Construindo índice FAISS...")
    index = build_faiss_index(embeddings.copy())

    index_path = out_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"[03] Índice FAISS salvo em: {index_path}")

    # Salva também os embeddings raw para debug/re-indexação
    emb_path = out_dir / "embeddings.npy"
    np.save(str(emb_path), embeddings)

    meta = {
        "model": model_name,
        "total_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "chunk_size_words": chunk_size,
        "overlap_words": overlap,
        "index_type": "IndexFlatIP (cosine)",
        "elapsed_sec": round(elapsed, 2),
    }
    (out_dir / "embedding_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[03] Metadados salvos em: {out_dir}/embedding_meta.json")

    # Libera VRAM
    del embed_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n[03] Step 03 concluído!")


def main():
    parser = argparse.ArgumentParser(description="OCR Markdown → Embeddings + FAISS")
    parser.add_argument("--md",          default="output/ocr_output.md",        help="Markdown de entrada")
    parser.add_argument("--out",         default="output/chunks",               help="Diretório de saída")
    parser.add_argument("--model",       default="Qwen/Qwen3-Embedding-4B",     help="Modelo de embeddings")
    parser.add_argument("--chunk-size",  type=int, default=400,                 help="Tamanho do chunk em palavras")
    parser.add_argument("--overlap",     type=int, default=80,                  help="Sobreposição entre chunks")
    parser.add_argument("--batch-size",  type=int, default=8,                   help="Batch size para embeddings")
    args = parser.parse_args()

    build_embeddings(
        md_path=Path(args.md),
        out_dir=Path(args.out),
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
