import os
import time
import warnings
import shutil
import json
from typing import List, Dict, Optional, Any, Callable, TypeVar, Tuple
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from dotenv import load_dotenv
from agents import function_tool
from functools import lru_cache
from tqdm import tqdm

load_dotenv()

# Retry em falhas de rede/SSL ao chamar a API de embeddings
_EMBEDDING_RETRY_MAX = 4
_EMBEDDING_RETRY_BASE_DELAY = 5.0

def _is_retryable_embedding_error(exc: BaseException) -> bool:
    """Considera erro de conexão/SSL como retentável."""
    msg = str(exc).lower()
    if "connection" in msg or "ssl" in msg or "read" in msg or "timeout" in msg:
        return True
    exc_cls = type(exc).__name__
    if "Connection" in exc_cls or "ReadError" in exc_cls or "Timeout" in exc_cls:
        return True
    return False

T = TypeVar("T")

def _retry_embedding_call(fn: Callable[[], T], desc: str = "embedding") -> T:
    """Executa fn e, em falha de rede/SSL, repete com backoff exponencial."""
    last: Optional[BaseException] = None
    for tentativa in range(_EMBEDDING_RETRY_MAX):
        try:
            return fn()
        except BaseException as e:
            last = e
            if tentativa == _EMBEDDING_RETRY_MAX - 1 or not _is_retryable_embedding_error(e):
                raise
            delay = _EMBEDDING_RETRY_BASE_DELAY * (2 ** tentativa)
            tqdm.write(f"    [RETRY] {desc}: {type(e).__name__} — nova tentativa em {delay:.0f}s...")
            time.sleep(delay)
    raise last  # type: ignore

# =============================================================================
# CONFIGURAÇÃO DE PERSISTÊNCIA DE VECTORSTORES
# =============================================================================

# Modelo de embeddings (Hugging Face Sentence Transformers - local, sem API key)
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_PROVIDER = "huggingface"
EMBEDDING_DIMENSION = 1024
VECTORSTORE_META_FILE = "embedding_meta.json"

# Diretório base para persistir os vectorstores
VECTORSTORE_PERSIST_DIR = os.path.join("vectorstores")

def _get_persist_directory(collection_name: str) -> str:
    """Retorna o caminho de persistência para uma coleção específica."""
    return os.path.join(VECTORSTORE_PERSIST_DIR, collection_name)

def _expected_embedding_metadata() -> Dict[str, Any]:
    """Retorna metadados esperados para o embedding atual."""
    return {
        "provider": EMBEDDING_PROVIDER,
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
    }

def _meta_file_path(persist_dir: str) -> str:
    """Retorna o caminho do arquivo de metadados do vectorstore."""
    return os.path.join(persist_dir, VECTORSTORE_META_FILE)

def _carregar_metadata_vectorstore(persist_dir: str) -> Optional[Dict[str, Any]]:
    """Lê metadados do vectorstore persistido."""
    meta_path = _meta_file_path(persist_dir)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _salvar_metadata_vectorstore(persist_dir: str) -> None:
    """Salva metadados do embedding usado para o vectorstore."""
    os.makedirs(persist_dir, exist_ok=True)
    meta_path = _meta_file_path(persist_dir)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_expected_embedding_metadata(), f, ensure_ascii=False, indent=2)

def _vectorstore_esta_compativel(collection_name: str) -> Tuple[bool, str]:
    """
    Verifica se existe vectorstore persistido e se ele é compatível com o embedding atual.
    Compatibilidade é baseada em metadados persistidos (provider/model/dimensão).
    """
    persist_dir = _get_persist_directory(collection_name)
    if not os.path.isdir(persist_dir):
        return False, "diretorio_nao_encontrado"

    metadata = _carregar_metadata_vectorstore(persist_dir)
    if metadata is None:
        return False, "metadata_ausente_ou_invalida"

    expected = _expected_embedding_metadata()
    if metadata != expected:
        return False, "embedding_incompativel"

    return True, "ok"

def _preparar_vectorstore_persistido(collection_name: str) -> bool:
    """
    Garante que o vectorstore persistido esteja compatível.
    Se existir incompatível, remove para permitir recriação limpa.
    """
    compativel, motivo = _vectorstore_esta_compativel(collection_name)
    if compativel:
        return True

    persist_dir = _get_persist_directory(collection_name)
    if os.path.isdir(persist_dir):
        print(
            f"[MIGRACAO] Vectorstore '{collection_name}' incompatível ({motivo}). "
            "Removendo para recriar com o embedding atual..."
        )
        limpar_vectorstore(collection_name)
    return False

def _carregar_vectorstore_persistido(collection_name: str) -> Chroma:
    """
    Carrega um vectorstore do disco.
    
    Args:
        collection_name: Nome da coleção
        
    Returns:
        VectorStore carregado do disco
    """
    persist_dir = _get_persist_directory(collection_name)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    
    return vectorstore

def limpar_vectorstore(collection_name: str) -> bool:
    """
    Remove um vectorstore persistido do disco.
    Útil para forçar recriação após atualização de documentos.
    
    Args:
        collection_name: Nome da coleção a ser removida
        
    Returns:
        True se removido com sucesso, False se não existia
    """
    persist_dir = _get_persist_directory(collection_name)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"[LIMPO] Vectorstore '{collection_name}' removido de {persist_dir}")
        return True
    return False

def limpar_todos_vectorstores() -> int:
    """
    Remove todos os vectorstores persistidos.
    
    Returns:
        Número de vectorstores removidos
    """
    if not os.path.exists(VECTORSTORE_PERSIST_DIR):
        return 0
    
    count = 0
    for item in os.listdir(VECTORSTORE_PERSIST_DIR):
        item_path = os.path.join(VECTORSTORE_PERSIST_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            count += 1
            print(f"[LIMPO] Removido: {item}")
    
    return count

def listar_vectorstores_persistidos() -> List[str]:
    """
    Lista todos os vectorstores que estão persistidos no disco.
    
    Returns:
        Lista de nomes de coleções persistidas
    """
    if not os.path.exists(VECTORSTORE_PERSIST_DIR):
        return []
    
    return [
        item for item in os.listdir(VECTORSTORE_PERSIST_DIR)
        if os.path.isdir(os.path.join(VECTORSTORE_PERSIST_DIR, item))
    ]

# Suprimir warnings de PDFs problemáticos
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")

class SafePDFLoader(PyPDFLoader):
    """
    Loader de PDF com tratamento de erros melhorado.
    Ignora páginas/arquivos com problemas de estrutura.
    """
    
    def load(self) -> List[Document]:
        """Carrega o PDF ignorando erros de estrutura."""
        try:
            return super().load()
        except Exception as e:
            # Imprime o caminho do PDF com problema e o tipo de erro
            print(f"    [PDF ERRO] {self.file_path}")
            print(f"               Motivo: {type(e).__name__}: {str(e)[:100]}")
            return []

# =============================================================================
# CONFIGURAÇÃO DE PASTAS POR TIPO DE AGENTE
# Baseado no Relatório de Classificação de Doutrinas - Arquitetura de Banco de Dados
# =============================================================================

BASE_DOUTRINAS_PATH = os.path.join("data", "Doutrinas (Markdown)")

# Pastas do Banco Geral - Conteúdo transversal utilizado por todas as áreas
PASTAS_BANCO_GERAL = [
    "CÓDIGOS-VADE MECUM E JURISPRUDÊNCIA",
    "DIREITO CONSTITUCIONAL",
    "DIREITOS HUMANOS",
    "INTRODUÇÃO AO ESTUDO DO DIREITO",
    "HERMENÊUTICA JURÍDICA",
    "FILOSOFIA DO DIREITO- etc",
    "METODOLOGIA-DICIONÁRIOS, PORTUGUÊS JURÍDICO E ETC",
    "ÉTICA PROFISSIONAL",
    "PRÁTICA JURÍDICA",
    "MANUAL DE RECURSOS",
    "PROCESSO CIVIL",
    "Juizados",
    "LGPD",
]

# Pastas adicionais do banco geral em data/ (não estão em Doutrinas (Markdown))
# Nestas pastas, indexamos apenas .txt e .pdf.
PASTAS_BANCO_GERAL_EXTRAS_DATA = [
    "banks",
    "informativos_tjdft",
]

# Pastas específicas por agente especialista
PASTAS_POR_AGENTE: Dict[str, List[str]] = {
    "tributario": [
        "DIREITO TRIBUTÁRIO",
        "DIREITO FINANCEIRO BRASILEIRO",
    ],
    "civil": [
        "DIREITO CIVIL",
        "DIREITO DE FAMÍLIA",
        "DIREITO DE FAMÍLIAS-ALIENAÇÃO PARENTAL",
        "Direito de Seguros",
        "DANO MORAL-ASSÉDIO MORAL",
    ],
    "penal": [
        "DIREITO PENAL",
        "PROCESSO PENAL",
        "PACOTE ANTICRIME",
        "LEI ABUSO DE AUTORIDADE",
        "MEDICINA LEGAL",
        "PENAL MILITAR",
    ],
    "trabalhista": [
        "DIREITO DO TRABALHO",
        "PROCESSO DO TRABALHO",
    ],
    "empresarial": [
        "DIREITO EMPRESARIAL-COMERCIAL",
        "ARBITRAGEM, MEDIAÇÃO E CONCILIAÇÃO",
    ],
    "administrativo": [
        "DIREITO ADMINISTRATIVO",
    ],
    "consumidor": [
        "DIREITO DO CONSUMIDOR",
    ],
    "ambiental": [
        "DIREITO AMBIENTAL",
    ],
    "previdenciario": [
        "DIREITO PREVIDENCIÁRIO",
    ],
    "eleitoral": [
        "DIREITO ELEITORAL",
    ],
    "internacional": [
        "INTERNACIONAL PRIVADO",
        "INTERNACIONAL PÚBLICO",
    ],
    "infancia": [
        "ECA",
    ],
    "transito": [
        "CTB",
    ],
    "medico": [
        "DIREITO MÉDICO",
    ],
}

# Material Complementar (opcional, pode ser incluído no banco geral)
PASTAS_COMPLEMENTARES = [
    "ANTROPOLOGIA",
    "SOCIOLOGIA",
    "PSICOLOGIA JURÍDICA",
    "COVID",
    "OBRAS ESTRANGEIRAS",
]

# Cache para armazenar retrievers já criados (singleton pattern)
# Usamos um dicionário mutável em nível de módulo
class _RetrieverCache:
    """Singleton para cache de retrievers, evitando duplicação em imports múltiplos."""
    _instance = None
    _cache: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get(self, key: str) -> Any:
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        self._cache[key] = value
    
    def has(self, key: str) -> bool:
        return key in self._cache

_retrievers_cache = _RetrieverCache()

def _listar_arquivos_recursivo(pasta: str, extensoes: List[str]) -> List[str]:
    """Lista todos os arquivos com as extensões especificadas em uma pasta recursivamente."""
    arquivos = []
    for root, dirs, files in os.walk(pasta):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in extensoes):
                arquivos.append(os.path.join(root, file))
    return arquivos

def _carregar_md_seguro(caminho: str) -> List[Document]:
    """
    Carrega um arquivo Markdown de forma segura (usa TextLoader com UTF-8).
    
    Args:
        caminho: Caminho completo do arquivo .md
        
    Returns:
        Lista de documentos ou lista vazia se houver erro
    """
    try:
        loader = TextLoader(caminho, encoding='utf-8')
        return loader.load()
    except Exception as e:
        print(f"    [MD ERRO] {caminho}")
        print(f"             Motivo: {type(e).__name__}: {str(e)[:80]}")
        return []

def _carregar_txt_seguro(caminho: str) -> List[Document]:
    """
    Carrega um arquivo TXT de forma segura (usa TextLoader com UTF-8).

    Args:
        caminho: Caminho completo do arquivo .txt

    Returns:
        Lista de documentos ou lista vazia se houver erro
    """
    try:
        loader = TextLoader(caminho, encoding="utf-8")
        return loader.load()
    except Exception as e:
        print(f"    [TXT ERRO] {caminho}")
        print(f"              Motivo: {type(e).__name__}: {str(e)[:80]}")
        return []

def _carregar_pdf_seguro(caminho: str) -> List[Document]:
    """
    Carrega um arquivo PDF de forma segura (usa SafePDFLoader).

    Args:
        caminho: Caminho completo do arquivo .pdf

    Returns:
        Lista de documentos ou lista vazia se houver erro
    """
    try:
        loader = SafePDFLoader(caminho)
        return loader.load()
    except Exception as e:
        print(f"    [PDF ERRO] {caminho}")
        print(f"              Motivo: {type(e).__name__}: {str(e)[:80]}")
        return []

def _carregar_documentos_de_pastas(
    pastas: List[str],
    base_path: str = BASE_DOUTRINAS_PATH,
    extensoes: Optional[List[str]] = None,
    mostrar_progresso: bool = True
) -> list:
    """
    Carrega documentos das subpastas dentro de base_path com as extensões informadas.
    Por padrão, carrega apenas .md (compatível com o comportamento anterior).
    
    Args:
        pastas: Lista de nomes de subpastas para carregar (dentro de base_path)
        base_path: Caminho base onde as pastas estão (ex.: data/Doutrinas (Markdown))
        extensoes: Lista de extensões permitidas (ex.: [".md"], [".txt", ".pdf"])
        mostrar_progresso: Se True, exibe barra de progresso
    
    Returns:
        Lista de documentos carregados
    """
    
    if extensoes is None:
        extensoes = [".md"]

    all_docs = []
    total_erros = 0
    
    # Barra de progresso para pastas
    pastas_iter = tqdm(pastas, desc="📁 Pastas", unit="pasta", disable=not mostrar_progresso)
    
    for pasta in pastas_iter:
        
        pasta_path = os.path.join(base_path, pasta)
        
        if not os.path.exists(pasta_path):
            tqdm.write(f"[AVISO] Pasta não encontrada: {pasta_path}")
            continue
        
        pastas_iter.set_postfix_str(f"{pasta[:30]}...")
        
        pasta_docs = []
        erros_pasta = 0
        
        todos_arquivos: List[Tuple[str, str]] = []
        for ext in extensoes:
            arquivos_ext = _listar_arquivos_recursivo(pasta_path, [ext])
            tipo = ext.lower().lstrip(".")
            todos_arquivos.extend((arquivo, tipo) for arquivo in arquivos_ext)
      
        if todos_arquivos:
            
            # Barra de progresso para arquivos dentro da pasta
            arquivos_iter = tqdm(
                todos_arquivos, 
                desc=f"  📄 {pasta[:25]}", 
                unit="arq",
                leave=False,
                disable=not mostrar_progresso
            )
            
            for arquivo, tipo in arquivos_iter:
                nome_arquivo = os.path.basename(arquivo)[:20]
                arquivos_iter.set_postfix_str(nome_arquivo)
                
                if tipo == "md":
                    docs = _carregar_md_seguro(arquivo)
                elif tipo == "txt":
                    docs = _carregar_txt_seguro(arquivo)
                elif tipo == "pdf":
                    docs = _carregar_pdf_seguro(arquivo)
                else:
                    continue
                
                if docs:
                    pasta_docs.extend(docs)
                else:
                    erros_pasta += 1
        
        total_erros += erros_pasta
        status_erro = f" ({erros_pasta} erros)" if erros_pasta > 0 else ""
        tqdm.write(f"    ✓ {pasta}: {len(pasta_docs)} docs{status_erro}")
        all_docs.extend(pasta_docs)
    
    if total_erros > 0:
        tqdm.write(f"\n  [RESUMO] Total de arquivos com erro: {total_erros}")
    
    return all_docs

def _criar_vectorstore(
    docs: list, 
    collection_name: str = "default", 
    mostrar_progresso: bool = True,
    persistir: bool = True
) -> any:
    """
    Cria um vectorstore a partir de uma lista de documentos.
    Se persistir=True, salva no disco para uso futuro.
    
    Args:
        docs: Lista de documentos para indexar
        collection_name: Nome da coleção no Chroma
        mostrar_progresso: Se True, exibe barra de progresso
        persistir: Se True, salva o vectorstore no disco
    
    Returns:
        VectorStore configurado
    """
    if not docs:
        raise ValueError("Nenhum documento fornecido para criar o vectorstore")
    
    # Definir diretório de persistência
    persist_directory = _get_persist_directory(collection_name) if persistir else None
    
    # Criar diretório se necessário
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
    
    # Fragmentação (Chunking) com progresso
    tqdm.write("  🔪 Fragmentando documentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    # Processar documentos com barra de progresso
    chunks = []
    with tqdm(total=len(docs), desc="  ✂️  Chunking", unit="doc", disable=not mostrar_progresso) as pbar:
        for doc in docs:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
            pbar.update(1)
            pbar.set_postfix(chunks=len(chunks))
    
    tqdm.write(f"  ✓ Fragmentados {len(chunks)} chunks")
    
    # Criar embeddings e vectorstore com progresso
    tqdm.write("  🧠 Criando embeddings (isso pode demorar)...")
    
    # Processar em batches para mostrar progresso
    batch_size = 500
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if len(chunks) <= batch_size:
        # Se poucos chunks, criar direto (com retry em falha de rede/SSL)
        with tqdm(total=1, desc="  🔗 Embeddings", unit="batch", disable=not mostrar_progresso) as pbar:
            vectorstore = _retry_embedding_call(
                lambda: Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                ),
                "from_documents",
            )
            pbar.update(1)
    else:
        # Processar em batches para chunks grandes (com retry em falha de rede/SSL)
        with tqdm(total=len(chunks), desc="  🔗 Embeddings", unit="chunk", disable=not mostrar_progresso) as pbar:
            primeiro_batch = chunks[:batch_size]
            vectorstore = _retry_embedding_call(
                lambda: Chroma.from_documents(
                    documents=primeiro_batch,
                    embedding=embeddings,
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                ),
                "from_documents",
            )
            pbar.update(len(primeiro_batch))

            for i in range(batch_size, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]

                def _add_batch():
                    vectorstore.add_documents(batch)

                _retry_embedding_call(_add_batch, "add_documents")
                pbar.update(len(batch))
    
    if persist_directory:
        tqdm.write(f"  💾 Vectorstore '{collection_name}' persistido em: {persist_directory}")
        _salvar_metadata_vectorstore(persist_directory)
    
    tqdm.write(f"  ✓ Vectorstore '{collection_name}' criado com {len(chunks)} chunks")
    
    return vectorstore

def criarRetrieverPorAgente(
    tipo_agente: str, 
    incluir_banco_geral: bool = True, 
    mostrar_progresso: bool = True,
    forcar_recriacao: bool = False
) -> any:
    """
    Cria um retriever específico para um tipo de agente, carregando apenas
    os documentos relevantes conforme definido na arquitetura.
    
    O vectorstore é persistido no disco. Em execuções futuras, o vectorstore
    será carregado do disco ao invés de ser recriado, economizando tempo e custos.
    
    Args:
        tipo_agente: Nome do agente (ex: "civil", "penal", "tributario")
        incluir_banco_geral: Se True, inclui também as pastas do banco geral
        mostrar_progresso: Se True, exibe barras de progresso
        forcar_recriacao: Se True, ignora vectorstore persistido e recria do zero
    
    Returns:
        Retriever configurado para o agente
    
    Raises:
        ValueError: Se o tipo de agente não for reconhecido
    """

    cache_key = f"{tipo_agente}_{'com_geral' if incluir_banco_geral else 'sem_geral'}"
    collection_name = f"agente_{tipo_agente.lower()}"
    
    # Verificar cache em memória
    if _retrievers_cache.has(cache_key):
        print(f"[CACHE] ⚡ Usando retriever em cache para '{tipo_agente}'")
        return _retrievers_cache.get(cache_key)
    
    tipo_agente_lower = tipo_agente.lower()
    
    if tipo_agente_lower not in PASTAS_POR_AGENTE:
        tipos_disponiveis = list(PASTAS_POR_AGENTE.keys())
        raise ValueError(
            f"Tipo de agente '{tipo_agente}' não reconhecido. "
            f"Tipos disponíveis: {tipos_disponiveis}"
        )
    
    # Verificar se existe vectorstore persistido no disco (e compatível com embedding atual)
    if not forcar_recriacao and _preparar_vectorstore_persistido(collection_name):
        
        print(f"\n{'='*60}")
        print(f"💾 CARREGANDO vectorstore persistido: {tipo_agente.upper()}")
        print(f"{'='*60}")
        
        vectorstore = _carregar_vectorstore_persistido(collection_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Armazenar em cache de memória
        _retrievers_cache.set(cache_key, retriever)
        
        print(f"✅ [OK] Retriever '{tipo_agente}' carregado do disco!")
        print(f"   📁 Local: {_get_persist_directory(collection_name)}")
        print(f"{'='*60}\n")
        
        return retriever
    
    # Se não existe ou forçou recriação, criar do zero
    if forcar_recriacao:
        limpar_vectorstore(collection_name)
    
    print(f"\n{'='*60}")
    print(f"🚀 Criando retriever para agente: {tipo_agente.upper()}")
    print(f"{'='*60}")
    
    # Definir etapas para progresso geral
    etapas = ["Carregar Docs", "Criar Vectorstore", "Configurar Retriever"]
    
    with tqdm(total=len(etapas), desc="📊 Progresso Geral", unit="etapa", disable=not mostrar_progresso) as progresso_geral:
        
        # Etapa 1: Carregar documentos
        progresso_geral.set_postfix_str("Carregando documentos...")
        
        # Determinar pastas a carregar
        pastas_a_carregar = []
        
        if incluir_banco_geral:
            tqdm.write(f"\n[1/3] 📚 Carregando Banco Geral ({len(PASTAS_BANCO_GERAL)} pastas)...")
            pastas_a_carregar.extend(PASTAS_BANCO_GERAL)
        
        pastas_especialista = PASTAS_POR_AGENTE[tipo_agente_lower]
        tqdm.write(f"[1/3] 📖 Carregando pastas específicas ({len(pastas_especialista)} pastas)...")
        pastas_a_carregar.extend(pastas_especialista)
        
        # Carregar documentos
        docs = _carregar_documentos_de_pastas(pastas_a_carregar, mostrar_progresso=mostrar_progresso)
        
        if not docs:
            raise ValueError(f"Nenhum documento encontrado para o agente '{tipo_agente}'")
        
        tqdm.write(f"\n✅ Total de documentos carregados: {len(docs)}")
        progresso_geral.update(1)
        
        # Etapa 2: Criar vectorstore (com persistência)
        progresso_geral.set_postfix_str("Criando vectorstore...")
        tqdm.write(f"\n[2/3] 🗄️  Criando vectorstore (com persistência)...")
        
        vectorstore = _criar_vectorstore(
            docs, 
            collection_name=collection_name, 
            mostrar_progresso=mostrar_progresso,
            persistir=True
        )

        progresso_geral.update(1)
        
        # Etapa 3: Configurar retriever
        progresso_geral.set_postfix_str("Configurando retriever...")
        tqdm.write(f"\n[3/3] 🔍 Configurando retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Armazenar em cache
        _retrievers_cache.set(cache_key, retriever)
        progresso_geral.update(1)
    
    print(f"\n✅ [OK] Retriever para '{tipo_agente}' criado e persistido!")
    print(f"   💾 Salvo em: {_get_persist_directory(collection_name)}")
    print(f"{'='*60}\n")
    
    return retriever

def criarRetrieverBancoGeral(
    mostrar_progresso: bool = True,
    forcar_recriacao: bool = False
) -> any:
    """
    Cria um retriever apenas com o banco geral (conteúdo transversal).
    Útil para consultas gerais que não se encaixam em nenhuma especialidade.
    
    O vectorstore é persistido no disco. Em execuções futuras, será carregado
    do disco ao invés de ser recriado, economizando tempo e custos.
    
    Args:
        mostrar_progresso: Se True, exibe barras de progresso
        forcar_recriacao: Se True, ignora vectorstore persistido e recria do zero
    
    Returns:
        Retriever configurado com o banco geral
    """
    
    cache_key = "banco_geral"
    collection_name = "banco_geral"
    
    # Verificar cache em memória
    if _retrievers_cache.has(cache_key):
        print(f"[CACHE] ⚡ Usando retriever do banco geral em cache")
        return _retrievers_cache.get(cache_key)
    
    # Verificar se existe vectorstore persistido no disco (e compatível com embedding atual)
    if not forcar_recriacao and _preparar_vectorstore_persistido(collection_name):
        print(f"\n{'='*60}")
        print(f"💾 CARREGANDO vectorstore persistido: BANCO GERAL")
        print(f"{'='*60}")
        
        vectorstore = _carregar_vectorstore_persistido(collection_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Armazenar em cache de memória
        _retrievers_cache.set(cache_key, retriever)
        
        print(f"✅ [OK] Retriever do banco geral carregado do disco!")
        print(f"   📁 Local: {_get_persist_directory(collection_name)}")
        print(f"{'='*60}\n")
        
        return retriever
    
    # Se não existe ou forçou recriação, criar do zero
    if forcar_recriacao:
        limpar_vectorstore(collection_name)
    
    print(f"\n{'='*60}")
    print("🚀 Criando retriever do BANCO GERAL")
    print(f"{'='*60}")
    
    etapas = ["Carregar Docs", "Criar Vectorstore", "Configurar Retriever"]
    
    with tqdm(total=len(etapas), desc="📊 Progresso Geral", unit="etapa", disable=not mostrar_progresso) as progresso_geral:
        
        # Etapa 1: Carregar documentos
        progresso_geral.set_postfix_str("Carregando documentos...")
        tqdm.write(f"\n[1/3] 📚 Carregando {len(PASTAS_BANCO_GERAL)} pastas .md do banco geral...")
        docs_md = _carregar_documentos_de_pastas(
            PASTAS_BANCO_GERAL,
            base_path=BASE_DOUTRINAS_PATH,
            extensoes=[".md"],
            mostrar_progresso=mostrar_progresso
        )
        tqdm.write(
            f"[1/3] 📚 Carregando {len(PASTAS_BANCO_GERAL_EXTRAS_DATA)} pastas extras (.txt/.pdf) em data/..."
        )
        docs_extras = _carregar_documentos_de_pastas(
            PASTAS_BANCO_GERAL_EXTRAS_DATA,
            base_path="data",
            extensoes=[".txt", ".pdf"],
            mostrar_progresso=mostrar_progresso
        )
        docs = docs_md + docs_extras
        
        if not docs:
            raise ValueError("Nenhum documento encontrado no banco geral")
        
        tqdm.write(f"\n✅ Total de documentos: {len(docs)}")
        progresso_geral.update(1)
        
        # Etapa 2: Criar vectorstore (com persistência)
        progresso_geral.set_postfix_str("Criando vectorstore...")
        tqdm.write(f"\n[2/3] 🗄️  Criando vectorstore (com persistência)...")
        vectorstore = _criar_vectorstore(
            docs, 
            collection_name=collection_name, 
            mostrar_progresso=mostrar_progresso,
            persistir=True
        )
        
        progresso_geral.update(1)
        
        # Etapa 3: Configurar retriever
        progresso_geral.set_postfix_str("Configurando retriever...")
        tqdm.write(f"\n[3/3] 🔍 Configurando retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        _retrievers_cache.set(cache_key, retriever)
        progresso_geral.update(1)
    
    print(f"\n✅ [OK] Retriever do banco geral criado e persistido!")
    print(f"   💾 Salvo em: {_get_persist_directory(collection_name)}")
    print(f"{'='*60}\n")
    
    return retriever

# Funções de conveniência para cada agente específico

def criarRetrieverTributario() -> any:
    """Cria retriever para o Agente Tributário."""
    return criarRetrieverPorAgente("tributario")

def criarRetrieverCivil() -> any:
    """Cria retriever para o Agente Civil."""
    return criarRetrieverPorAgente("civil")

def criarRetrieverPenal() -> any:
    """Cria retriever para o Agente Penal."""
    return criarRetrieverPorAgente("penal")

def criarRetrieverTrabalhista() -> any:
    """Cria retriever para o Agente Trabalhista."""
    return criarRetrieverPorAgente("trabalhista")

def criarRetrieverEmpresarial() -> any:
    """Cria retriever para o Agente Empresarial."""
    return criarRetrieverPorAgente("empresarial")

def criarRetrieverAdministrativo() -> any:
    """Cria retriever para o Agente Administrativo."""
    return criarRetrieverPorAgente("administrativo")

def criarRetrieverConsumidor() -> any:
    """Cria retriever para o Agente Consumidor."""
    return criarRetrieverPorAgente("consumidor")

def criarRetrieverAmbiental() -> any:
    """Cria retriever para o Agente Ambiental."""
    return criarRetrieverPorAgente("ambiental")

def criarRetrieverPrevidenciario() -> any:
    """Cria retriever para o Agente Previdenciário."""
    return criarRetrieverPorAgente("previdenciario")

def criarRetrieverEleitoral() -> any:
    """Cria retriever para o Agente Eleitoral."""
    return criarRetrieverPorAgente("eleitoral")

def criarRetrieverInternacional() -> any:
    """Cria retriever para o Agente Internacional."""
    return criarRetrieverPorAgente("internacional")

def criarRetrieverInfancia() -> any:
    """Cria retriever para o Agente de Infância e Juventude (ECA)."""
    return criarRetrieverPorAgente("infancia")

def criarRetrieverTransito() -> any:
    """Cria retriever para o Agente de Trânsito (CTB)."""
    return criarRetrieverPorAgente("transito")

def criarRetrieverMedico() -> any:
    """Cria retriever para o Agente de Saúde/Médico."""
    return criarRetrieverPorAgente("medico")

def criarRetriever(mostrar_progresso: bool = True):
    """
    Função legada para compatibilidade.
    Cria um retriever com TODOS os documentos do diretório data.
    
    Args:
        mostrar_progresso: Se True, exibe barras de progresso
    
    ATENÇÃO: Prefira usar criarRetrieverPorAgente() para melhor performance
    e relevância dos resultados.
    """
    # Verificar cache
    cache_key = "legado_todos"
    if _retrievers_cache.has(cache_key):
        print(f"[CACHE] ⚡ Usando retriever legado em cache")
        return _retrievers_cache.get(cache_key)
    
    path = r"data"

    print(f"\n{'='*60}")
    print(f"🚀 [LEGADO] Carregando TODOS os documentos de {path}")
    print(f"{'='*60}")

    etapas = ["Carregar TXT", "Carregar PDF", "Fragmentar", "Embeddings"]
    
    with tqdm(total=len(etapas), desc="📊 Progresso Geral", unit="etapa", disable=not mostrar_progresso) as progresso_geral:
        # Etapa 1: Carregar TXT
        progresso_geral.set_postfix_str("Carregando TXT...")
        tqdm.write("\n[1/4] 📄 Carregando arquivos .txt...")
        
        txt_loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        txt_docs = txt_loader.load()
        tqdm.write(f"    ✓ {len(txt_docs)} arquivos .txt carregados")
        progresso_geral.update(1)

        # Etapa 2: Carregar PDF
        progresso_geral.set_postfix_str("Carregando PDF...")
        tqdm.write("\n[2/4] 📕 Carregando arquivos .pdf...")
        
        pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        tqdm.write(f"    ✓ {len(pdf_docs)} arquivos .pdf carregados")
        progresso_geral.update(1)

        # Combinar documentos
        docs = txt_docs + pdf_docs
        tqdm.write(f"\n✅ Total: {len(docs)} documentos carregados")

        # Etapa 3: Fragmentação
        progresso_geral.set_postfix_str("Fragmentando...")
        tqdm.write("\n[3/4] ✂️  Fragmentando documentos...")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = []
        
        with tqdm(total=len(docs), desc="  ✂️  Chunking", unit="doc", leave=False, disable=not mostrar_progresso) as pbar:
            for doc in docs:
                doc_chunks = text_splitter.split_documents([doc])
                chunks.extend(doc_chunks)
                pbar.update(1)
                pbar.set_postfix(chunks=len(chunks))
        
        tqdm.write(f"    ✓ {len(chunks)} chunks criados")
        progresso_geral.update(1)

        # Etapa 4: Embeddings e Vectorstore
        progresso_geral.set_postfix_str("Criando embeddings...")
        tqdm.write("\n[4/4] 🧠 Criando embeddings e vectorstore...")
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        batch_size = 500
        
        if len(chunks) <= batch_size:
            with tqdm(total=1, desc="  🔗 Embeddings", unit="batch", leave=False, disable=not mostrar_progresso) as pbar:
                vectorstore = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings
                )
                pbar.update(1)
        else:
            with tqdm(total=len(chunks), desc="  🔗 Embeddings", unit="chunk", leave=False, disable=not mostrar_progresso) as pbar:
                primeiro_batch = chunks[:batch_size]
                vectorstore = Chroma.from_documents(
                    documents=primeiro_batch,
                    embedding=embeddings
                )
                pbar.update(len(primeiro_batch))
                
                for i in range(batch_size, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    vectorstore.add_documents(batch)
                    pbar.update(len(batch))
        
        tqdm.write(f"    ✓ Vectorstore criado com {len(chunks)} chunks")
        progresso_geral.update(1)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    _retrievers_cache.set(cache_key, retriever)
    
    print(f"\n✅ [OK] Retriever legado criado com sucesso!")
    print(f"{'='*60}\n")

    return retriever

def criar_tool_buscar_documentos(retriever):
    """
    Registra a função como uma tool para o Agents SDK.
    
    Args:
        retriever: Retriever configurado para busca
    
    Returns:
        Tool function para buscar documentos
    """

    @function_tool
    async def buscar_documentos(query: str) -> str:
        """Busca informações relevantes nos documentos jurídicos indexados."""
        resultados = retriever.invoke(query)
        textos = [doc.page_content for doc in resultados]
        if not textos:
            return "Nenhum trecho relevante encontrado nos documentos."
        return "\n\n---\n\n".join(textos[:3])

    return buscar_documentos

def listar_agentes_disponiveis() -> List[str]:
    """
    Retorna a lista de tipos de agentes disponíveis.
    
    Returns:
        Lista de nomes de agentes
    """
    return list(PASTAS_POR_AGENTE.keys())

def obter_pastas_por_agente(tipo_agente: str) -> Dict[str, List[str]]:
    """
    Retorna as pastas configuradas para um tipo de agente.
    
    Args:
        tipo_agente: Nome do agente
    
    Returns:
        Dicionário com pastas 'geral' e 'especialista'
    """
    tipo_agente_lower = tipo_agente.lower()
    
    if tipo_agente_lower not in PASTAS_POR_AGENTE:
        raise ValueError(f"Tipo de agente '{tipo_agente}' não reconhecido")
    
    return {
        "geral": PASTAS_BANCO_GERAL,
        "especialista": PASTAS_POR_AGENTE[tipo_agente_lower]
    }