from __future__ import annotations

"""
Utilitário para (re)construir o índice Chroma usado pelo retriever.

Uso (Windows / PowerShell):
  .\\.venv\\Scripts\\python.exe -m app.agents.tools.build_vectorstore

Vars de ambiente:
  - RAG_DATA_DIR: diretório base dos documentos (default: ./data)
  - RAG_PERSIST_DIR: onde persistir o Chroma (default: ./vectorstore/chroma)
  - RAG_INCLUDE_PDF: 1 para incluir PDFs (default: 0)
  - RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP
"""

import os

from .retriever import _build_vectorstore, _get_data_dir, _get_persist_dir, _persist_dir_has_index

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

def main() -> int:
    data_dir = _get_data_dir()
    persist_dir = _get_persist_dir()

    print(f"RAG_DATA_DIR     = {data_dir}")
    print(f"RAG_PERSIST_DIR  = {persist_dir}")
    print(f"RAG_INCLUDE_PDF  = {os.getenv('RAG_INCLUDE_PDF', '0')}")
    print(f"RAG_CHUNK_SIZE   = {os.getenv('RAG_CHUNK_SIZE', '1000')}")
    print(f"RAG_CHUNK_OVERLAP= {os.getenv('RAG_CHUNK_OVERLAP', '100')}")

    already = _persist_dir_has_index(persist_dir)
    if already:
        print("ATENCAO: ja existe conteudo no diretorio de persistencia. Ele sera reutilizado/atualizado.")

    print("Construindo indice... (isso pode levar alguns minutos e consumir embeddings)")
    _build_vectorstore(data_dir=data_dir, persist_dir=persist_dir)
    print("OK: indice Chroma pronto.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

