"""
04_qa_pipeline.py
------------------
Pipeline RAG completo:
  1. Carrega chunks + índice FAISS (step 03)
  2. Para cada uma das 10 perguntas de avaliação:
     a. Gera embedding da pergunta (Qwen3-Embedding-4B)
     b. Recupera os top-k chunks mais relevantes
     c. Monta prompt com contexto + pergunta
     d. Gera resposta com Qwen/Qwen3.5-9B
     e. Formata resposta com citação exata (página, autor, linhas, trecho)
  3. Salva tudo em output/answers.md (Markdown formatado)
  4. Salva também output/answers.json (estruturado)

Uso:
    python pipeline/04_qa_pipeline.py [--chunks output/chunks]
                                      [--embed-model Qwen/Qwen3-Embedding-4B]
                                      [--gen-model Qwen/Qwen3.5-9B]
                                      [--top-k 5]
                                      [--out output]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import numpy as np
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        GenerationConfig,
    )
except ImportError:
    sys.exit("[ERRO] transformers/torch/numpy não instalados.")

try:
    import faiss
except ImportError:
    sys.exit("[ERRO] faiss não instalado.")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================================
# 10 Perguntas de Avaliação — Fonseca: Introdução Teórica à História do Direito
# ============================================================================
QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "Q01",
        "tema": "Conceito e objeto",
        "pergunta": (
            "Qual é o conceito de história do direito proposto por Fonseca e qual é "
            "o objeto central dessa disciplina segundo o livro?"
        ),
    },
    {
        "id": "Q02",
        "tema": "Metodologia historiográfica",
        "pergunta": (
            "Quais são as principais abordagens metodológicas para o estudo da "
            "história do direito apresentadas pelo autor?"
        ),
    },
    {
        "id": "Q03",
        "tema": "Direito e poder",
        "pergunta": (
            "Como Fonseca analisa a relação entre direito e poder na perspectiva "
            "histórica? Que autores ou escolas ele mobiliza para fundamentar essa relação?"
        ),
    },
    {
        "id": "Q04",
        "tema": "Direito romano",
        "pergunta": (
            "Qual é a relevância do direito romano para a formação dos ordenamentos "
            "jurídicos modernos segundo o livro?"
        ),
    },
    {
        "id": "Q05",
        "tema": "Modernidade e positivismo",
        "pergunta": (
            "Como o liberalismo jurídico e o positivismo influenciaram a concepção "
            "moderna de direito segundo a análise de Fonseca?"
        ),
    },
    {
        "id": "Q06",
        "tema": "Dogmática jurídica",
        "pergunta": (
            "O que o autor entende por 'dogmática jurídica' e quais são suas "
            "limitações históricas apontadas no livro?"
        ),
    },
    {
        "id": "Q07",
        "tema": "Direito brasileiro",
        "pergunta": (
            "Como se deu a formação e a recepção do direito europeu no contexto "
            "jurídico brasileiro segundo a perspectiva histórica do livro?"
        ),
    },
    {
        "id": "Q08",
        "tema": "Nova história do direito",
        "pergunta": (
            "Quais são as principais críticas da chamada 'Nova História do Direito' "
            "à historiografia jurídica tradicional apresentadas por Fonseca?"
        ),
    },
    {
        "id": "Q09",
        "tema": "Interdisciplinaridade",
        "pergunta": (
            "De que maneira o autor articula a história do direito com outras "
            "disciplinas como filosofia, sociologia e antropologia jurídica?"
        ),
    },
    {
        "id": "Q10",
        "tema": "Papel do historiador",
        "pergunta": (
            "Qual é a concepção de Fonseca sobre o papel do historiador do direito "
            "na sociedade contemporânea e por que essa perspectiva importa?"
        ),
    },
]


# ============================================================================
# Embedding helpers (reusados do step 03)
# ============================================================================

EMBED_INSTRUCTION = "Represent this query for retrieval:"


def _last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden.shape[0]
    return last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]


def embed_query(
    query: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    instruction: str = EMBED_INSTRUCTION,
    max_length: int = 256,
) -> np.ndarray:
    """Retorna embedding L2-normalizado de uma consulta."""
    text = f"{instruction}\n{query}" if instruction else query
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.inference_mode():
        outputs = model(**encoded)

    emb = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.cpu().float().numpy()


# ============================================================================
# Retrieval
# ============================================================================

def retrieve(
    query_emb: np.ndarray,
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Retorna os top_k chunks mais similares com scores."""
    q = query_emb.copy()
    faiss.normalize_L2(q)
    scores, ids = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = dict(chunks[idx])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results


# ============================================================================
# Prompt de geração
# ============================================================================

SYSTEM_PROMPT = textwrap.dedent("""\
    Você é um especialista em história do direito. Responda à pergunta do usuário
    **exclusivamente com base nos trechos fornecidos** do livro "Introdução Teórica
    à História do Direito" de Ricardo Marcelo Fonseca.

    Regras:
    - Cite explicitamente a página, o autor e o trecho relevante em cada ponto da resposta.
    - Use o formato: (Fonseca, p. X — linhas Y-Z: "trecho literal curto")
    - Se os trechos fornecidos não forem suficientes para responder com segurança,
      diga isso claramente.
    - Responda em português do Brasil.
    - Seja preciso, acadêmico e direto.
""").strip()


def build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(
            f"[Trecho {i} | Página {chunk['page']} | "
            f"Linhas {chunk['start_line']}-{chunk['end_line']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def build_prompt(question: str, context: str, use_chat_template: bool = True) -> str:
    if use_chat_template:
        # Para modelos de instrução, retornamos tupla (system, user)
        return question  # será usado via chat template
    # Fallback plain prompt
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"TRECHOS DO LIVRO:\n{context}\n\n"
        f"PERGUNTA:\n{question}\n\n"
        f"RESPOSTA:"
    )


# ============================================================================
# Geração
# ============================================================================

def generate_answer(
    question: str,
    context: str,
    gen_model: AutoModelForCausalLM,
    gen_tokenizer: AutoTokenizer,
    max_new_tokens: int = 1024,
) -> str:
    """Gera resposta usando chat template se disponível."""
    device = next(gen_model.parameters()).device

    # Tenta chat template (modelos instruct)
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TRECHOS DO LIVRO:\n{context}\n\n"
                    f"PERGUNTA:\n{question}"
                ),
            },
        ]
        inputs_text = gen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        use_chat = True
    except Exception:
        inputs_text = build_prompt(question, context, use_chat_template=False)
        use_chat = False

    encoded = gen_tokenizer(inputs_text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.inference_mode():
        out_ids = gen_model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
            pad_token_id=gen_tokenizer.eos_token_id,
        )

    new_ids = out_ids[0][encoded["input_ids"].shape[1]:]
    answer = gen_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return answer


# ============================================================================
# Formatação da saída Markdown
# ============================================================================

def format_answer_md(
    q_meta: Dict[str, str],
    answer: str,
    retrieved: List[Dict[str, Any]],
    q_index: int,
    total: int,
) -> str:
    lines = [
        f"## Pergunta {q_meta['id']} — {q_meta['tema']}",
        "",
        f"**Pergunta:** {q_meta['pergunta']}",
        "",
        "---",
        "",
        "### Resposta",
        "",
        answer,
        "",
        "---",
        "",
        "### Fontes Recuperadas (RAG)",
        "",
    ]

    for i, chunk in enumerate(retrieved, 1):
        score_pct = round(chunk["retrieval_score"] * 100, 1)
        trecho = chunk["text"][:200].replace("\n", " ") + ("..." if len(chunk["text"]) > 200 else "")
        lines += [
            f"**[Fonte {i}]** — Score: {score_pct}%",
            f"- **Autor:** {chunk['author']}",
            f"- **Obra:** {chunk['title']}",
            f"- **Página:** {chunk['page']}",
            f"- **Linhas:** {chunk['start_line']}–{chunk['end_line']}",
            f"- **Trecho:** \"{trecho}\"",
            "",
        ]

    lines.append(f"*Pergunta {q_index} de {total}*")
    return "\n".join(lines)


# ============================================================================
# Pipeline principal
# ============================================================================

def run_qa(
    chunks_dir: Path,
    out_dir: Path,
    embed_model_name: str = "Qwen/Qwen3-Embedding-4B",
    gen_model_name: str = "Qwen/Qwen3.5-9B",
    top_k: int = 5,
    max_new_tokens: int = 1024,
) -> None:
    chunks_path = chunks_dir / "chunks.json"
    index_path  = chunks_dir / "faiss.index"

    if not chunks_path.exists():
        sys.exit(f"[ERRO] {chunks_path} não encontrado. Execute o step 03.")
    if not index_path.exists():
        sys.exit(f"[ERRO] {index_path} não encontrado. Execute o step 03.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Carrega artefatos do step 03
    # ------------------------------------------------------------------
    print(f"\n[04] Carregando chunks: {chunks_path}")
    chunks: List[Dict[str, Any]] = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"[04] Chunks carregados: {len(chunks)}")

    print(f"[04] Carregando índice FAISS: {index_path}")
    index = faiss.read_index(str(index_path))
    print(f"[04] Vetores no índice: {index.ntotal}")

    # ------------------------------------------------------------------
    # Modelo de embeddings (query side)
    # ------------------------------------------------------------------
    print(f"\n[04] Carregando tokenizer de embeddings: {embed_model_name}")
    emb_tokenizer = AutoTokenizer.from_pretrained(embed_model_name, trust_remote_code=True)
    print(f"[04] Carregando modelo de embeddings...")
    emb_model = AutoModel.from_pretrained(
        embed_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    emb_model.eval()
    print(f"[04] Embed model device: {next(emb_model.parameters()).device}")

    # ------------------------------------------------------------------
    # Modelo de geração (Qwen3.5-9B)
    # ------------------------------------------------------------------
    print(f"\n[04] Carregando tokenizer de geração: {gen_model_name}")
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=True)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    print(f"[04] Carregando modelo de geração (pode levar alguns minutos)...")
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    gen_model.eval()
    print(f"[04] Gen model device: {next(gen_model.parameters()).device}")

    # ------------------------------------------------------------------
    # Processa cada pergunta
    # ------------------------------------------------------------------
    print(f"\n[04] Processando {len(QUESTIONS)} perguntas com top_k={top_k}...\n")

    all_answers = []
    md_blocks   = [
        "# Avaliação RAG — Introdução Teórica à História do Direito\n",
        "**Autor da obra:** Ricardo Marcelo Fonseca  ",
        "**Modelo OCR:** zai-org/GLM-OCR  ",
        f"**Modelo de embeddings:** {embed_model_name}  ",
        f"**Modelo de geração:** {gen_model_name}  ",
        f"**Top-k chunks:** {top_k}  ",
        f"**Data:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "\n---\n",
    ]

    q_iter = enumerate(QUESTIONS, 1)
    if tqdm:
        q_iter = tqdm(list(q_iter), desc="Perguntas", unit="q")

    for q_idx, q_meta in q_iter:
        question = q_meta["pergunta"]
        print(f"\n[04] {q_meta['id']}: {q_meta['tema']}")

        # a) Embedding da pergunta
        t0 = time.time()
        q_emb = embed_query(question, emb_model, emb_tokenizer)

        # b) Retrieval
        retrieved = retrieve(q_emb, index, chunks, top_k=top_k)

        # c) Contexto
        context = build_context(retrieved)

        # d) Geração
        answer = generate_answer(
            question=question,
            context=context,
            gen_model=gen_model,
            gen_tokenizer=gen_tokenizer,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        print(f"      Tempo: {elapsed:.1f}s | Resposta: {len(answer)} chars")

        # e) Formata
        md_block = format_answer_md(q_meta, answer, retrieved, q_idx, len(QUESTIONS))
        md_blocks.append(md_block)
        md_blocks.append("\n---\n")

        # Estruturado para JSON
        all_answers.append({
            "id":        q_meta["id"],
            "tema":      q_meta["tema"],
            "pergunta":  question,
            "resposta":  answer,
            "fontes": [
                {
                    "rank":          i + 1,
                    "score":         round(c["retrieval_score"], 4),
                    "chunk_id":      c["chunk_id"],
                    "pagina":        c["page"],
                    "autor":         c["author"],
                    "titulo":        c["title"],
                    "linha_inicial": c["start_line"],
                    "linha_final":   c["end_line"],
                    "trecho":        c["text"][:400],
                }
                for i, c in enumerate(retrieved)
            ],
            "elapsed_sec": round(elapsed, 2),
        })

    # ------------------------------------------------------------------
    # Salva resultados
    # ------------------------------------------------------------------
    md_path = out_dir / "answers.md"
    md_path.write_text("\n".join(md_blocks), encoding="utf-8")
    print(f"\n[04] Markdown salvo em: {md_path}")

    json_path = out_dir / "answers.json"
    json_path.write_text(
        json.dumps(all_answers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[04] JSON salvo em: {json_path}")

    # Libera VRAM
    del gen_model, gen_tokenizer, emb_model, emb_tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n[04] Pipeline completo! {len(QUESTIONS)} respostas geradas.")
    print(f"[04] Abra {md_path} para ver os resultados.")


def main():
    parser = argparse.ArgumentParser(description="RAG: 10 perguntas sobre Fonseca")
    parser.add_argument("--chunks",       default="output/chunks",             help="Diretório com chunks + FAISS")
    parser.add_argument("--embed-model",  default="Qwen/Qwen3-Embedding-4B",   help="Modelo de embeddings")
    parser.add_argument("--gen-model",    default="Qwen/Qwen3.5-9B",           help="Modelo de geração")
    parser.add_argument("--top-k",        type=int, default=5,                 help="Chunks recuperados por pergunta")
    parser.add_argument("--max-tokens",   type=int, default=1024,              help="max_new_tokens na geração")
    parser.add_argument("--out",          default="output",                    help="Diretório de saída")
    args = parser.parse_args()

    run_qa(
        chunks_dir=Path(args.chunks),
        out_dir=Path(args.out),
        embed_model_name=args.embed_model,
        gen_model_name=args.gen_model,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
