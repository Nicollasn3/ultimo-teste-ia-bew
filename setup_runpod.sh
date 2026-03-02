#!/bin/bash
# setup_runpod.sh — Configura o ambiente RunPod (GPU A40/CUDA) para o RAG Fonseca
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Setup RunPod: Qwen3-Embedding-4B + Ministral-3:8b          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# ── 1. Detectar versão CUDA e instalar torch compatível ──────────────────────
echo "[1/4] Detectando CUDA..."

CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
echo "      CUDA detectado: ${CUDA_VER:-não encontrado}"
echo "      Python: $(python3 --version)"
echo "      pip: $(pip3 --version)"

# Remove torch/torchvision instalados (versões incompatíveis)
echo "      Removendo torch/torchvision antigos..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Instala versões compatíveis conforme CUDA  (sem --quiet para ver erros)
if [[ "$CUDA_VER" == "12."* ]]; then
    echo "      Instalando torch 2.3.1 + torchvision 0.18.1 para CUDA 12.1..."
    pip install \
        torch==2.3.1 \
        torchvision==0.18.1 \
        torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VER" == "11.8"* ]]; then
    echo "      Instalando torch 2.3.1 + torchvision 0.18.1 para CUDA 11.8..."
    pip install \
        torch==2.3.1 \
        torchvision==0.18.1 \
        torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu118
else
    echo "      AVISO: CUDA não detectado — instalando build CPU..."
    pip install torch==2.3.1 torchvision==0.18.1
fi

# Verifica se torch foi instalado corretamente
echo ""
echo "      Verificando torch..."
python3 -c "
import torch
print(f'      torch     : {torch.__version__}')
print(f'      CUDA ok   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'      GPU       : {torch.cuda.get_device_name(0)}')
    print(f'      VRAM      : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
"

# ── 2. Instalar dependências Python ──────────────────────────────────────────
echo ""
echo "[2/4] Instalando dependências Python..."
pip install \
    "transformers>=4.51.0" \
    "huggingface_hub>=0.20" \
    "numpy" \
    "requests" \
    "accelerate"

# Verifica transformers
python3 -c "import transformers; print(f'      transformers: {transformers.__version__}')"
echo "      OK"

# ── 3. Configurar cache HuggingFace em storage persistente (se disponível) ───
echo ""
echo "[3/4] Configurando paths..."

# /runpod-volume é o storage persistente entre reinicializações do RunPod
if [ -d "/runpod-volume" ]; then
    HF_CACHE="/runpod-volume/hf_cache"
    echo "      Storage persistente detectado → HF_HOME=$HF_CACHE"
else
    HF_CACHE="/workspace/hf_cache"
    echo "      Sem /runpod-volume → HF_HOME=$HF_CACHE (temporário)"
fi

mkdir -p "$HF_CACHE"

# Salva .env para o script Python carregar
WORKSPACE_DIR=$(dirname "$(realpath "$0")")
cat > "$WORKSPACE_DIR/.env_rag" << EOF
export HF_HOME=$HF_CACHE
export HUGGINGFACE_HUB_CACHE=$HF_CACHE
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=ministral-3:8b
EOF

echo "      .env_rag salvo em: $WORKSPACE_DIR/.env_rag"

# ── 4. Verificar / Instalar Ollama e baixar modelo ───────────────────────────
echo ""
echo "[4/4] Configurando Ollama..."

if ! command -v ollama &> /dev/null; then
    echo "      Instalando Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "      Ollama já instalado: $(ollama --version 2>/dev/null || echo 'ok')"
fi

# Inicia Ollama em background se não estiver rodando
if ! curl -sf "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo "      Iniciando servidor Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
fi

echo "      Baixando ministral-3:8b (~6 GB)..."
ollama pull ministral-3:8b

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Setup concluído!                                           ║"
echo "║                                                              ║"
echo "║  Para rodar (no mesmo diretório do script):                 ║"
echo "║    source .env_rag                                          ║"
echo "║    ollama serve &                                            ║"
echo "║    python3 test_rag_fonseca.py                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
