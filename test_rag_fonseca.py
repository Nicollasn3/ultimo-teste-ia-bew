#!/usr/bin/env python3
"""
test_rag_fonseca.py
===================
Pipeline RAG de teste para o livro:
  FONSECA, Ricardo Marcelo — Introdução Teórica à História do Direito

Embedding  : Qwen/Qwen3-Embedding-4B  (HuggingFace, roda localmente)
LLM        : ministral:8b via Ollama  (http://localhost:11434)
Vectorstore: coseno similarity em memória + cache numpy em disco

Uso:
  python test_rag_fonseca.py            # roda as 10 perguntas de teste
  python test_rag_fonseca.py --rebuild  # força recriar embeddings
"""

import os
import sys
import json
import time
import math
import glob
import textwrap
import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ─── CONFIGURAÇÃO ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

# Localiza o arquivo automaticamente (encoding do path é complicado no macOS)
def _encontrar_documento() -> Path:
    """Encontra o arquivo Fonseca em qualquer subdiretório."""
    # Tenta na raiz do projeto primeiro (cópia que pode existir)
    candidatos_raiz = list(BASE_DIR.glob("FONSECA*.md"))
    if candidatos_raiz:
        return candidatos_raiz[0]
    # Tenta na pasta de dados
    padrao = str(BASE_DIR / "data" / "Doutrinas (Markdown)" / "ANTROPOLOGIA" / "FONSECA*.md")
    candidatos_data = glob.glob(padrao)
    if candidatos_data:
        return Path(candidatos_data[0])
    raise FileNotFoundError(
        "Arquivo Fonseca não encontrado. Verifique se o .md está em:\n"
        f"  {BASE_DIR}/  ou  {BASE_DIR}/data/Doutrinas (Markdown)/ANTROPOLOGIA/"
    )

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "ministral-3:8b")

CHUNK_SIZE      = 800    # chars por chunk
CHUNK_OVERLAP   = 120    # overlap entre chunks
TOP_K           = 5      # chunks recuperados por query
EMBED_BATCH     = 4      # chunks por batch no embedding
MAX_TOKENS_LM   = 512    # max tokens na resposta do LLM

CACHE_DIR  = BASE_DIR / ".cache_embeddings"
CACHE_FILE = CACHE_DIR / "fonseca_qwen3_4b.npz"

# ─── 10 PERGUNTAS DE TESTE ────────────────────────────────────────────────────

PERGUNTAS: List[str] = [
    "O que é a história do direito e por que ela é importante segundo Fonseca?",
    "Qual é a crítica de Fonseca ao positivismo jurídico e ao historicismo?",
    "Qual é a relação entre teoria e metodologia na história do direito?",
    "Quem é Paolo Grossi e qual é sua influência na obra de Fonseca?",
    "O que é a 'Nouvelle Vague' da historiografia jurídica brasileira?",
    "Como a École des Annales influenciou a historiografia jurídica?",
    "Qual é a influência de Michel Foucault na análise histórica do direito?",
    "Como as 'Teses sobre o conceito de história' de Walter Benjamin se relacionam com a historiografia jurídica?",
    "Qual foi o papel do direito colonial e das Ordenações Filipinas na formação do direito brasileiro?",
    "Como Fonseca define o método histórico-crítico para o estudo do direito?",
]

# ─── CHUNKING ─────────────────────────────────────────────────────────────────

def chunk_texto(texto: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Divide o texto em chunks com overlap, tentando respeitar parágrafos.
    Ignora linhas muito curtas (artefatos de OCR).
    """
    # Pré-processamento: remove linhas com menos de 20 chars (ruído de OCR)
    linhas = texto.split("\n")
    linhas_limpas = []
    for linha in linhas:
        l = linha.strip()
        if len(l) >= 20 or l == "":
            linhas_limpas.append(l)
    texto_limpo = "\n".join(linhas_limpas)

    # Divide em parágrafos
    paragrafos = [p.strip() for p in texto_limpo.split("\n\n") if p.strip()]

    chunks: List[str] = []
    buffer = ""

    for paragrafo in paragrafos:
        candidato = buffer + " " + paragrafo if buffer else paragrafo
        if len(candidato) <= chunk_size:
            buffer = candidato
        else:
            if buffer:
                chunks.append(buffer.strip())
            # Paragrafo maior que chunk_size: divide por força bruta
            if len(paragrafo) > chunk_size:
                for start in range(0, len(paragrafo), chunk_size - overlap):
                    trecho = paragrafo[start : start + chunk_size]
                    if trecho.strip():
                        chunks.append(trecho.strip())
                buffer = paragrafo[-(overlap):]
            else:
                buffer = paragrafo

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if len(c) > 50]  # descarta mini-chunks

# ─── EMBEDDING (Qwen3-Embedding-4B) ──────────────────────────────────────────

def _escolher_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pooling pelo último token (recomendado para Qwen3-Embedding).
    Suporta padding tanto à esquerda quanto à direita.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]

def carregar_modelo_embedding(device: str) -> Tuple[AutoTokenizer, AutoModel]:
    """Carrega o tokenizer e o modelo Qwen3-Embedding-4B."""
    print(f"[EMBED] Carregando {EMBEDDING_MODEL} em '{device}'...")
    print("        (primeira vez pode demorar — ~8 GB de download)")

    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        trust_remote_code=True,
        padding_side="right",   # Qwen3 usa right-padding
    )
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)
    model.eval()
    print(f"[EMBED] Modelo carregado. Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model

def encode(
    textos: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    instrucao: str = "",
    batch_size: int = EMBED_BATCH,
) -> np.ndarray:
    """
    Codifica uma lista de textos em embeddings normalizados.

    Para queries (instrucao != ""): usa instrução de tarefa antes do texto.
    Para documentos (instrucao == ""): texto puro.
    """
    prefixo = instrucao if instrucao else ""
    todos_embeddings: List[np.ndarray] = []

    for i in range(0, len(textos), batch_size):
        lote = textos[i : i + batch_size]
        lote_prefixado = [prefixo + t for t in lote] if prefixo else lote

        encoded = tokenizer(
            lote_prefixado,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            saida = model(**encoded)
            emb = _last_token_pool(saida.last_hidden_state, encoded["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        todos_embeddings.append(emb.cpu().float().numpy())
        print(f"  lote {i // batch_size + 1}/{math.ceil(len(textos) / batch_size)}", end="\r")

    print()
    return np.concatenate(todos_embeddings, axis=0)

# ─── VECTORSTORE EM MEMÓRIA ───────────────────────────────────────────────────

class VectorStore:
    """Vectorstore simples em memória com busca coseno via numpy."""

    def __init__(self, chunks: List[str], embeddings: np.ndarray):
        self.chunks = chunks
        # embeddings já devem estar normalizados (L2 = 1)
        self.embeddings = embeddings.astype(np.float32)

    def buscar(self, query_emb: np.ndarray, k: int = TOP_K) -> List[Tuple[float, str]]:
        """Retorna os k chunks mais similares à query."""
        q = query_emb.astype(np.float32).flatten()
        q /= np.linalg.norm(q) + 1e-10  # normaliza por segurança
        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunks[i]) for i in top_idx]

# ─── CACHE DE EMBEDDINGS ──────────────────────────────────────────────────────

def _hash_arquivo(path: Path) -> str:
    h = hashlib.md5()
    h.update(str(path.stat().st_mtime).encode())
    h.update(str(path.stat().st_size).encode())
    h.update(EMBEDDING_MODEL.encode())
    h.update(str(CHUNK_SIZE).encode())
    h.update(str(CHUNK_OVERLAP).encode())
    return h.hexdigest()[:16]

def salvar_cache(chunks: List[str], embeddings: np.ndarray, hash_doc: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_FILE,
        embeddings=embeddings,
        chunks=np.array(chunks, dtype=object),
        hash_doc=np.array([hash_doc]),
    )
    print(f"[CACHE] Embeddings salvos em {CACHE_FILE}")

def carregar_cache(hash_doc: str) -> Tuple[List[str], np.ndarray] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        dados = np.load(CACHE_FILE, allow_pickle=True)
        if dados["hash_doc"][0] != hash_doc:
            print("[CACHE] Documento ou configuração mudou → reconstruindo embeddings")
            return None
        chunks = list(dados["chunks"])
        embeddings = dados["embeddings"]
        print(f"[CACHE] Carregados {len(chunks)} chunks do cache ({CACHE_FILE.name})")
        return chunks, embeddings
    except Exception as e:
        print(f"[CACHE] Erro ao ler cache: {e} → reconstruindo")
        return None

# ─── OLLAMA LLM ───────────────────────────────────────────────────────────────

def verificar_ollama() -> bool:
    """Verifica se o Ollama está rodando e se o modelo está disponível."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        modelos = [m["name"] for m in r.json().get("models", [])]
        print(f"[OLLAMA] Modelos disponíveis: {modelos}")
        if not any(OLLAMA_MODEL in m for m in modelos):
            print(f"\n[AVISO] Modelo '{OLLAMA_MODEL}' não encontrado!")
            print(f"        Execute: ollama pull {OLLAMA_MODEL}\n")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(f"\n[ERRO] Ollama não está rodando em {OLLAMA_BASE_URL}")
        print("       Inicie o Ollama com: ollama serve")
        print(f"       Depois baixe o modelo: ollama pull {OLLAMA_MODEL}\n")
        return False

def gerar_resposta(pergunta: str, contexto: List[str]) -> str:
    """Chama o Ollama para gerar uma resposta com base no contexto recuperado."""
    contexto_txt = "\n\n---\n\n".join(
        [f"[Trecho {i+1}]\n{c}" for i, c in enumerate(contexto)]
    )
    prompt = f"""Você é um assistente especializado em história do direito.
Use SOMENTE as informações dos trechos abaixo para responder à pergunta.
Se a informação não estiver nos trechos, diga "Não encontrei essa informação nos trechos recuperados."

TRECHOS DO LIVRO:
{contexto_txt}

PERGUNTA: {pergunta}

RESPOSTA (em português, máximo 3 parágrafos):"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS_LM,
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[ERRO NA GERAÇÃO: {e}]"

# ─── RELATÓRIO DE ACURÁCIA ────────────────────────────────────────────────────

def imprimir_separador(char: str = "─", largura: int = 72) -> None:
    print(char * largura)

def imprimir_resultado(
    idx: int,
    pergunta: str,
    chunks_recuperados: List[Tuple[float, str]],
    resposta: str,
    tempo_retrieval: float,
    tempo_geracao: float,
) -> None:
    imprimir_separador("═")
    print(f"PERGUNTA {idx+1}/{len(PERGUNTAS)}")
    print(f"  {pergunta}")
    imprimir_separador()

    print(f"CHUNKS RECUPERADOS (top {TOP_K}) — retrieval em {tempo_retrieval:.2f}s:")
    for rank, (score, chunk) in enumerate(chunks_recuperados, 1):
        preview = textwrap.shorten(chunk, width=120, placeholder="...")
        print(f"  [{rank}] score={score:.4f} | {preview}")

    imprimir_separador()
    print(f"RESPOSTA DO MODELO ({OLLAMA_MODEL}) — geração em {tempo_geracao:.2f}s:")
    for linha in textwrap.wrap(resposta, width=72):
        print(f"  {linha}")
    print()

# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="RAG test: Qwen3-Embedding-4B + Mistral")
    parser.add_argument("--rebuild", action="store_true", help="Força recriar embeddings mesmo se há cache")
    parser.add_argument("--no-llm",  action="store_true", help="Só testa retrieval, sem chamar o LLM")
    parser.add_argument("--k",       type=int, default=TOP_K, help=f"Número de chunks a recuperar (default: {TOP_K})")
    args = parser.parse_args()

    k = args.k

    print("\n" + "═" * 72)
    print(" RAG TEST — Fonseca: Introdução Teórica à História do Direito")
    print(f" Embedding : {EMBEDDING_MODEL}")
    print(f" LLM       : {OLLAMA_MODEL} via Ollama")
    print("═" * 72 + "\n")

    # 1. Verificar LLM
    if not args.no_llm:
        if not verificar_ollama():
            print("[INFO] Rodando só com retrieval (--no-llm). Use --no-llm para suprimir este aviso.")
            args.no_llm = True

    # 2. Carregar documento
    doc_path = _encontrar_documento()
    print(f"[DOC] Carregando: {doc_path.name} ({doc_path.stat().st_size // 1024} KB)")
    texto = doc_path.read_text(encoding="utf-8", errors="replace")
    print(f"[DOC] Tamanho: {len(texto):,} chars")

    hash_doc = _hash_arquivo(doc_path)

    # 3. Carregar ou criar embeddings
    cached = None if args.rebuild else carregar_cache(hash_doc)

    if cached:
        chunks, embeddings = cached
    else:
        # 3a. Chunking
        print("\n[CHUNK] Dividindo documento...")
        chunks = chunk_texto(texto)
        print(f"[CHUNK] {len(chunks)} chunks criados (tamanho médio: {sum(len(c) for c in chunks)//len(chunks)} chars)")

        # 3b. Embedding
        device = _escolher_device()
        tokenizer, model = carregar_modelo_embedding(device)

        print(f"\n[EMBED] Codificando {len(chunks)} chunks (isso pode demorar)...")
        t0 = time.time()
        embeddings = encode(chunks, tokenizer, model, device, instrucao="")
        t1 = time.time()
        print(f"[EMBED] Pronto em {t1-t0:.1f}s — shape: {embeddings.shape}")

        # Libera memória do modelo de embedding
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        salvar_cache(chunks, embeddings, hash_doc)

    # 4. Criar vectorstore
    vs = VectorStore(chunks, embeddings)

    # 5. Encoding das queries (carrega o modelo só se precisar)
    print("\n[RETRIEVAL] Codificando perguntas de teste...")
    device = _escolher_device()

    # Carrega o modelo novamente para as queries (ou mantém em memória se não foi deletado)
    # (Se não criamos embeddings acima, o modelo ainda não foi carregado)
    if cached:
        tokenizer, model = carregar_modelo_embedding(device)

    instrucao_query = (
        "Instruct: Recuperar trechos do livro que respondam à pergunta.\nQuery: "
    )
    query_embeddings = encode(PERGUNTAS, tokenizer, model, device, instrucao=instrucao_query, batch_size=2)
    print(f"[RETRIEVAL] {len(PERGUNTAS)} queries codificadas")

    del model  # libera VRAM/RAM

    # 6. Rodar as 10 perguntas
    print(f"\n[TESTE] Executando {len(PERGUNTAS)} perguntas...\n")
    resultados = []

    for i, (pergunta, q_emb) in enumerate(zip(PERGUNTAS, query_embeddings)):
        # Retrieval
        t_ret_inicio = time.time()
        recuperados = vs.buscar(q_emb, k=k)
        t_ret_fim = time.time()
        tempo_retrieval = t_ret_fim - t_ret_inicio

        # Geração
        if not args.no_llm:
            t_gen_inicio = time.time()
            contexto_textos = [c for _, c in recuperados]
            resposta = gerar_resposta(pergunta, contexto_textos)
            t_gen_fim = time.time()
            tempo_geracao = t_gen_fim - t_gen_inicio
        else:
            resposta = "[LLM desabilitado — use sem --no-llm para ver respostas]"
            tempo_geracao = 0.0

        imprimir_resultado(i, pergunta, recuperados, resposta, tempo_retrieval, tempo_geracao)

        resultados.append({
            "pergunta": pergunta,
            "top_score": recuperados[0][0] if recuperados else 0.0,
            "tempo_retrieval_s": round(tempo_retrieval, 4),
            "tempo_geracao_s": round(tempo_geracao, 2),
            "resposta_chars": len(resposta),
        })

    # 7. Sumário de performance
    imprimir_separador("═")
    print("SUMÁRIO DE PERFORMANCE")
    imprimir_separador()
    print(f"{'#':<4} {'Score Top-1':>12} {'Retrieval':>10} {'Geração':>10}  Pergunta")
    imprimir_separador()
    for i, r in enumerate(resultados):
        pergunta_curta = textwrap.shorten(r["pergunta"], width=45, placeholder="...")
        print(
            f"{i+1:<4} {r['top_score']:>12.4f} {r['tempo_retrieval_s']:>9.4f}s "
            f"{r['tempo_geracao_s']:>9.2f}s  {pergunta_curta}"
        )
    imprimir_separador()

    scores = [r["top_score"] for r in resultados]
    print(f"\n  Score médio top-1    : {sum(scores)/len(scores):.4f}")
    print(f"  Score mínimo top-1   : {min(scores):.4f}")
    print(f"  Score máximo top-1   : {max(scores):.4f}")

    if not args.no_llm:
        t_ger = [r["tempo_geracao_s"] for r in resultados]
        print(f"  Tempo médio geração  : {sum(t_ger)/len(t_ger):.2f}s")

    print(f"\n  Total de chunks no índice: {len(chunks)}")
    print(f"  Dimensão dos embeddings  : {embeddings.shape[1]}")
    print()

    # 8. Salvar resultados JSON
    json_path = BASE_DIR / "resultados_rag_fonseca.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo_embedding": EMBEDDING_MODEL,
                "modelo_llm": OLLAMA_MODEL,
                "documento": doc_path.name,
                "num_chunks": len(chunks),
                "embedding_dim": int(embeddings.shape[1]),
                "top_k": k,
                "resultados": resultados,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[SALVO] Resultados em: {json_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
