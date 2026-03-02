"""
02_ocr_pages.py
----------------
Roda o modelo zai-org/GLM-OCR em cada imagem PNG gerada pelo step 01.

Saída:
  output/ocr_output.md       — texto completo (todas as páginas)
  output/ocr_pages/          — um .txt por página
  output/ocr_log.json        — log com tempo por página

Uso:
    python pipeline/02_ocr_pages.py [--pages output/pages] [--out output] [--model zai-org/GLM-OCR]
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError:
    sys.exit("[ERRO] transformers/torch não instalados. Execute setup_runpod.sh")

# Desabilita o limite de tamanho de imagem do Pillow (PDFs de alta resolução
# podem exceder o limite padrão de ~178MP que o PIL usa como proteção anti-DoS).
# Seguro aqui porque as imagens são geradas localmente pelo step 01.
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    pass

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# Pós-processamento: limpa tokens especiais do GLM-OCR
# ---------------------------------------------------------------------------
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>|<s>|</s>")


def clean_ocr_text(text: str) -> str:
    """Remove tokens especiais e normaliza espaços."""
    text = _SPECIAL_TOKEN_RE.sub("", text)
    # colapsa múltiplas linhas em branco em no máximo 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Inferência GLM-OCR (uma imagem por vez para economizar VRAM)
# ---------------------------------------------------------------------------

def run_glm_ocr(
    pages_dir: Path,
    out_dir: Path,
    model_name: str = "zai-org/GLM-OCR",
    max_new_tokens: int = 8192,
    batch_size: int = 1,          # GLM-OCR é pesado; batch=1 é o padrão seguro
) -> None:
    """Processa todas as páginas PNG com GLM-OCR."""

    pages_meta_path = pages_dir / "metadata.json"
    if not pages_meta_path.exists():
        sys.exit(f"[ERRO] metadata.json não encontrado em {pages_dir}. Execute o step 01.")

    meta = json.loads(pages_meta_path.read_text(encoding="utf-8"))
    page_files = sorted(pages_dir.glob("page_*.png"))

    if not page_files:
        sys.exit(f"[ERRO] Nenhum PNG encontrado em {pages_dir}")

    # Diretórios de saída
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_out = out_dir / "ocr_pages"
    pages_out.mkdir(exist_ok=True)

    print(f"\n[02] Modelo: {model_name}")
    print(f"[02] Páginas: {len(page_files)}")

    # ------------------------------------------------------------------
    # Carrega modelo e processor
    # ------------------------------------------------------------------
    print("[02] Carregando processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    print("[02] Carregando modelo (isso pode levar alguns minutos)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"[02] Modelo carregado em: {device}")

    # ------------------------------------------------------------------
    # Processa cada página
    # ------------------------------------------------------------------
    all_pages_text = []
    log_entries = []

    iterator = tqdm(page_files, desc="OCR", unit="pág") if tqdm else page_files

    for img_path in iterator:
        page_num = int(img_path.stem.split("_")[1])
        t0 = time.time()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": str(img_path)},
                {"type": "text",  "text": "Text Recognition:"},
            ],
        }]

        inputs = None
        generated_ids = None
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # GLM-OCR não usa token_type_ids
            inputs.pop("token_type_ids", None)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            # Decodifica apenas os tokens novos (exclui o prompt)
            new_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
            raw_text = processor.decode(new_ids, skip_special_tokens=False)
            page_text = clean_ocr_text(raw_text)

        except Exception as exc:
            print(f"\n[02][WARN] Falha na página {page_num}: {exc}")
            page_text = f"[OCR FALHOU: {exc}]"

        finally:
            # Libera VRAM independente de sucesso ou falha
            if inputs is not None:
                del inputs
            if generated_ids is not None:
                del generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed = time.time() - t0

        # ---- Salva .txt individual
        page_txt_path = pages_out / f"page_{page_num:04d}.txt"
        page_txt_path.write_text(page_text, encoding="utf-8")

        # ---- Acumula para o markdown final
        page_block = (
            f"## Página {page_num}\n\n"
            f"<!-- source: page={page_num} author=\"Ricardo Marcelo Fonseca\" "
            f"title=\"Introdução Teórica à História do Direito\" -->\n\n"
            f"{page_text}"
        )
        all_pages_text.append((page_num, page_block))

        log_entries.append({
            "page": page_num,
            "chars": len(page_text),
            "elapsed_sec": round(elapsed, 2),
            "file": str(page_txt_path),
        })

    # ------------------------------------------------------------------
    # Monta markdown final na ordem das páginas
    # ------------------------------------------------------------------
    all_pages_text.sort(key=lambda x: x[0])
    author = meta.get("author", "Ricardo Marcelo Fonseca")
    title  = meta.get("title",  "Introdução Teórica à História do Direito")

    header = (
        f"# {title}\n\n"
        f"**Autor:** {author}  \n"
        f"**Modelo OCR:** {model_name}  \n"
        f"**Total de páginas:** {len(all_pages_text)}\n\n"
        "---\n\n"
    )

    full_md = header + "\n\n---\n\n".join(block for _, block in all_pages_text)
    md_path = out_dir / "ocr_output.md"
    md_path.write_text(full_md, encoding="utf-8")

    # Salva log
    log_path = out_dir / "ocr_log.json"
    log_path.write_text(
        json.dumps({"model": model_name, "pages": log_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_chars = sum(e["chars"] for e in log_entries)
    total_time  = sum(e["elapsed_sec"] for e in log_entries)
    print(f"\n[02] Concluído: {len(page_files)} páginas | {total_chars:,} chars | {total_time:.0f}s total")
    print(f"[02] Markdown salvo em: {md_path}")

    # Libera modelo
    del model, processor
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    parser = argparse.ArgumentParser(description="Imagens PNG → OCR Markdown (GLM-OCR)")
    parser.add_argument("--pages",  default="output/pages",      help="Diretório com os PNGs")
    parser.add_argument("--out",    default="output",             help="Diretório de saída")
    parser.add_argument("--model",  default="zai-org/GLM-OCR",   help="Modelo HuggingFace")
    parser.add_argument("--max-tokens", type=int, default=8192,  help="max_new_tokens por página")
    args = parser.parse_args()

    run_glm_ocr(
        pages_dir=Path(args.pages),
        out_dir=Path(args.out),
        model_name=args.model,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
