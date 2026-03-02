#!/bin/bash
# =============================================================================
# run.sh — Orquestrador completo do pipeline no RunPod
#
# Etapas:
#   1) PDF → imagens PNG (PyMuPDF, 300 DPI)
#   2) Imagens → OCR Markdown (GLM-OCR)
#   3) Markdown → Embeddings + FAISS (Qwen3-Embedding-4B)
#   4) RAG + 10 perguntas → respostas com citações (Qwen3.5-9B)
#
# Uso:
#   bash run.sh                  # roda tudo
#   bash run.sh --from 2         # recomeça do step 2
#   bash run.sh --only 4         # roda apenas o step 4
#   bash run.sh --pdf outro.pdf  # usa PDF diferente
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------
# Configurações (edite aqui se necessário)
# --------------------------------------------------------------------------
PDF="data/fonseca.pdf"
OCR_MODEL="zai-org/GLM-OCR"
EMBED_MODEL="Qwen/Qwen3-Embedding-4B"
GEN_MODEL="Qwen/Qwen3.5-9B"
DPI=300
CHUNK_SIZE=400
OVERLAP=80
TOP_K=5
MAX_TOKENS=1024
PAGES_DIR="output/pages"
OUT_DIR="output"
CHUNKS_DIR="output/chunks"

# --------------------------------------------------------------------------
# Cores
# --------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${GREEN}[RUN]${NC} $1"; }
step()  { echo -e "\n${BOLD}${CYAN}━━━ $1 ━━━${NC}"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
die()   { echo -e "${RED}[ERRO]${NC} $1"; exit 1; }

# --------------------------------------------------------------------------
# Argumentos
# --------------------------------------------------------------------------
FROM_STEP=1
ONLY_STEP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)  FROM_STEP="$2"; shift 2 ;;
        --only)  ONLY_STEP="$2"; shift 2 ;;
        --pdf)   PDF="$2";       shift 2 ;;
        *) warn "Argumento desconhecido: $1"; shift ;;
    esac
done

should_run() {
    local step_num="$1"
    if [[ "$ONLY_STEP" -gt 0 ]]; then
        [[ "$ONLY_STEP" -eq "$step_num" ]]
    else
        [[ "$step_num" -ge "$FROM_STEP" ]]
    fi
}

# --------------------------------------------------------------------------
# Verifica pré-requisitos
# --------------------------------------------------------------------------
PYTHON=$(command -v python3 || command -v python || die "Python não encontrado")

if [ ! -f "$PDF" ]; then
    # Tenta encontrar o PDF original
    ORIGINAL="FONSECA, Ricardo Marcelo. Introdução Teórica à História do Direito.pdf"
    if [ -f "$ORIGINAL" ]; then
        mkdir -p data
        cp "$ORIGINAL" "$PDF"
        log "PDF copiado para $PDF"
    else
        die "PDF não encontrado: $PDF\nColoque o PDF em $PDF ou passe --pdf <caminho>"
    fi
fi

mkdir -p output data

# --------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Pipeline GLM-OCR + Qwen RAG — Fonseca Direito      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
log "PDF:         $PDF"
log "OCR:         $OCR_MODEL"
log "Embeddings:  $EMBED_MODEL"
log "Geração:     $GEN_MODEL"
log "Iniciando em: step $FROM_STEP"
echo ""

# ===========================================================================
# STEP 1 — PDF → Imagens PNG
# ===========================================================================
if should_run 1; then
    step "Step 1/4 — PDF → Imagens PNG (${DPI} DPI)"
    "$PYTHON" pipeline/01_pdf_to_images.py \
        --pdf   "$PDF"      \
        --dpi   "$DPI"      \
        --out   "$PAGES_DIR"
    log "Step 1 concluído."
else
    log "Step 1 pulado."
fi

# ===========================================================================
# STEP 2 — OCR com GLM-OCR
# ===========================================================================
if should_run 2; then
    step "Step 2/4 — OCR das páginas (GLM-OCR)"
    if [ ! -f "$PAGES_DIR/metadata.json" ]; then
        die "metadata.json ausente em $PAGES_DIR. Execute o step 1 primeiro."
    fi
    "$PYTHON" pipeline/02_ocr_pages.py \
        --pages    "$PAGES_DIR"  \
        --out      "$OUT_DIR"    \
        --model    "$OCR_MODEL"
    log "Step 2 concluído."
else
    log "Step 2 pulado."
fi

# ===========================================================================
# STEP 3 — Embeddings + FAISS
# ===========================================================================
if should_run 3; then
    step "Step 3/4 — Embeddings + Índice FAISS (${EMBED_MODEL})"
    if [ ! -f "$OUT_DIR/ocr_output.md" ]; then
        die "ocr_output.md ausente em $OUT_DIR. Execute o step 2 primeiro."
    fi
    "$PYTHON" pipeline/03_build_embeddings.py \
        --md         "$OUT_DIR/ocr_output.md" \
        --out        "$CHUNKS_DIR"            \
        --model      "$EMBED_MODEL"           \
        --chunk-size "$CHUNK_SIZE"            \
        --overlap    "$OVERLAP"
    log "Step 3 concluído."
else
    log "Step 3 pulado."
fi

# ===========================================================================
# STEP 4 — RAG: 10 perguntas + respostas com citações
# ===========================================================================
if should_run 4; then
    step "Step 4/4 — RAG: 10 Perguntas (${GEN_MODEL})"
    if [ ! -f "$CHUNKS_DIR/faiss.index" ]; then
        die "faiss.index ausente em $CHUNKS_DIR. Execute o step 3 primeiro."
    fi
    "$PYTHON" pipeline/04_qa_pipeline.py \
        --chunks      "$CHUNKS_DIR"   \
        --embed-model "$EMBED_MODEL"  \
        --gen-model   "$GEN_MODEL"    \
        --top-k       "$TOP_K"        \
        --max-tokens  "$MAX_TOKENS"   \
        --out         "$OUT_DIR"
    log "Step 4 concluído."
else
    log "Step 4 pulado."
fi

# --------------------------------------------------------------------------
# Resumo final
# --------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  Pipeline concluído com sucesso!             ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}OCR Markdown:${NC}  $OUT_DIR/ocr_output.md"
echo -e "  ${CYAN}Respostas:${NC}     $OUT_DIR/answers.md      ← abra este!"
echo -e "  ${CYAN}Dados JSON:${NC}    $OUT_DIR/answers.json"
echo -e "  ${CYAN}Chunks:${NC}        $CHUNKS_DIR/chunks.json"
echo ""
