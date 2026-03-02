"""
01_pdf_to_images.py
--------------------
Converte cada página do PDF em uma imagem PNG de alta qualidade.

Saída:  output/pages/page_XXXX.png
        output/pages/metadata.json  (número de páginas, resolução, etc.)

Uso:
    python pipeline/01_pdf_to_images.py [--pdf data/fonseca.pdf] [--dpi 300] [--out output/pages]
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError:
    sys.exit("[ERRO] pymupdf não encontrado. Execute: pip install pymupdf")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def tqdm_or_range(iterable, **kwargs):
    if tqdm:
        return tqdm(iterable, **kwargs)
    return iterable


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 300) -> dict:
    """Renderiza cada página do PDF como PNG e salva em out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    print(f"[01] PDF: {pdf_path.name}")
    print(f"[01] Páginas: {n_pages} | DPI: {dpi}")
    print(f"[01] Saída: {out_dir}/")

    t0 = time.time()
    saved = []
    pages_iter = tqdm_or_range(range(n_pages), desc="Renderizando", unit="pág")

    for i in pages_iter:
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = out_dir / f"page_{i + 1:04d}.png"
        pix.save(str(img_path))
        saved.append({
            "page": i + 1,
            "file": str(img_path),
            "width_px": pix.width,
            "height_px": pix.height,
        })

    doc.close()
    elapsed = time.time() - t0

    metadata = {
        "pdf": str(pdf_path),
        "author": "Ricardo Marcelo Fonseca",
        "title": "Introdução Teórica à História do Direito",
        "total_pages": n_pages,
        "dpi": dpi,
        "elapsed_sec": round(elapsed, 2),
        "pages": saved,
    }

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[01] Concluído: {n_pages} páginas em {elapsed:.1f}s")
    print(f"[01] Metadados salvos em {meta_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="PDF → imagens PNG")
    parser.add_argument("--pdf", default="data/fonseca.pdf", help="Caminho do PDF")
    parser.add_argument("--dpi", type=int, default=300, help="Resolução (padrão: 300)")
    parser.add_argument("--out", default="output/pages", help="Diretório de saída")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"[ERRO] PDF não encontrado: {pdf_path}")

    pdf_to_images(pdf_path, Path(args.out), dpi=args.dpi)


if __name__ == "__main__":
    main()
