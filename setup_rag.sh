#!/bin/bash
# setup_rag.sh — Instala dependências e baixa o modelo Ollama para o RAG Fonseca
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Setup RAG: Qwen3-Embedding-4B + Ministral 8B via Ollama    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# ── 1. Instalar dependências Python ──────────────────────────────────────────
echo "[1/3] Instalando dependências Python..."
pip3 install --upgrade \
    "transformers>=4.51.0" \
    "torch>=2.0" \
    "numpy" \
    "requests" \
    "huggingface_hub"

echo "      OK"

# ── 2. Verificar/instalar Ollama ──────────────────────────────────────────────
echo ""
echo "[2/3] Verificando Ollama..."

if command -v ollama &> /dev/null; then
    echo "      Ollama já instalado: $(ollama --version 2>/dev/null || echo 'versão desconhecida')"
else
    echo "      Ollama não encontrado. Instalando via brew..."
    if command -v brew &> /dev/null; then
        brew install ollama
    else
        echo "      ERRO: brew não encontrado."
        echo "      Instale manualmente: https://ollama.com/download"
        exit 1
    fi
fi

# ── 3. Baixar modelo Mistral ──────────────────────────────────────────────────
echo ""
echo "[3/3] Baixando modelo '${OLLAMA_MODEL:-ministral:8b}' (~5 GB)..."
echo "      (pode demorar dependendo da conexão)"
echo ""

MODELO="${OLLAMA_MODEL:-ministral-3:8b}"

# Inicia o servidor Ollama em background se não estiver rodando
if ! curl -sf "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo "      Iniciando servidor Ollama em background..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
fi

ollama pull "$MODELO"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Setup concluído!                                           ║"
echo "║                                                              ║"
echo "║  Para rodar o teste:                                        ║"
echo "║    1. Em um terminal: ollama serve                          ║"
echo "║    2. Neste terminal: python3 test_rag_fonseca.py           ║"
echo "║                                                              ║"
echo "║  Na primeira execução, o Qwen3-Embedding-4B (~8 GB)         ║"
echo "║  será baixado automaticamente pelo HuggingFace.             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
