#!/bin/bash
# =============================================================================
# setup_runpod.sh — Instala todas as dependências do pipeline no RunPod
# =============================================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

log "=== Pipeline GLM-OCR + Qwen RAG — Setup RunPod ==="
log "Data: $(date)"

# --------------------------------------------------------------------------
# Verifica CUDA
# --------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    log "GPU detectada:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    warn "nvidia-smi não encontrado — rodando em CPU (muito lento para OCR)"
fi

PYTHON=$(command -v python3 || command -v python || die "Python não encontrado")
log "Python: $($PYTHON --version)"

PIP="$PYTHON -m pip"

# --------------------------------------------------------------------------
# Atualiza pip e instala pacotes base
# --------------------------------------------------------------------------
log "Atualizando pip..."
$PIP install --upgrade pip --quiet

log "Instalando dependências base..."
$PIP install --upgrade \
    pymupdf \
    pillow \
    tqdm \
    numpy \
    --quiet

log "Instalando PyTorch (CUDA 12.1)..."
$PIP install --upgrade \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

log "Instalando Transformers e utilitários..."
$PIP install --upgrade \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    einops \
    tokenizers \
    --quiet

log "Instalando FAISS (GPU)..."
$PIP install faiss-gpu --quiet 2>/dev/null || {
    warn "faiss-gpu falhou, instalando faiss-cpu..."
    $PIP install faiss-cpu --quiet
}

log "Instalando utilitários de texto..."
$PIP install --upgrade \
    tiktoken \
    regex \
    --quiet

# --------------------------------------------------------------------------
# Verifica torch + CUDA
# --------------------------------------------------------------------------
log "Verificando torch + CUDA..."
$PYTHON - <<'PYEOF'
import torch
print(f"  torch   : {torch.__version__}")
print(f"  CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PYEOF

# --------------------------------------------------------------------------
# Cria estrutura de diretórios
# --------------------------------------------------------------------------
log "Criando estrutura de diretórios..."
mkdir -p pipeline data output/pages output/chunks

# Move o PDF para data/ se ainda estiver na raiz
PDF_SRC="FONSECA, Ricardo Marcelo. Introdução Teórica à História do Direito.pdf"
if [ -f "$PDF_SRC" ] && [ ! -f "data/fonseca.pdf" ]; then
    cp "$PDF_SRC" data/fonseca.pdf
    log "PDF copiado para data/fonseca.pdf"
elif [ -f "data/fonseca.pdf" ]; then
    log "PDF já está em data/fonseca.pdf"
else
    warn "PDF não encontrado em '$PDF_SRC'. Coloque o PDF em data/fonseca.pdf manualmente."
fi

log "=== Setup concluído! ==="
log "Execute: bash run.sh"
