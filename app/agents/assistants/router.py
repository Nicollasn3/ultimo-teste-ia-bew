import os
import sys
from typing import Any, Dict, List, Optional, cast

# Permite rodar este arquivo diretamente (ex.: `python app/agents/assistants/router.py`)
# colocando a RAIZ do projeto no sys.path (para importar `app.*`) sem sombrear o pacote externo `agents`.
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from pydantic import BaseModel
from agents import Agent, Runner, handoff, RunContextWrapper
from app.agents.assistants.civil import CivilAgent
from app.agents.assistants.empresarial import EmpresarialAgent
from app.agents.assistants.penal import PenalAgent
from app.agents.assistants.trabalhista import TrabalhistaAgent
from app.agents.assistants.tributario import TributarioAgent
from app.agents.prompts.prompts import ROUTER_HANDOFF_INSTRUCTIONS
from app.agents.shared.base_agent import BaseAgent
from app.agents.utils.trace import trace
import asyncio
from dotenv import load_dotenv
import uuid
from app.agents.tools.memory import get_session_state, save_memory, set_active_agent
from app.agents.tools.retriever import criar_tool_buscar_documentos

load_dotenv()

class SpecialistHandoffData(BaseModel):
    reason: str = ""

class Router(BaseAgent):
    """
    Agente Router (triagem jurídica).
    
    Responsável por:
    - Entender o pedido do usuário
    - Coletar informações essenciais
    - Encaminhar para o agente especialista adequado
    
    Args:
        retriever_router: Retriever para o Router (banco geral). Opcional.
        retriever_civil: Retriever para o agente Civil. Opcional.
        retriever_penal: Retriever para o agente Penal. Opcional.
        retriever_trabalhista: Retriever para o agente Trabalhista. Opcional.
        retriever_tributario: Retriever para o agente Tributário. Opcional.
        retriever_empresarial: Retriever para o agente Empresarial. Opcional.
        session_id: ID da sessão para persistência de memória.
    """
    
    def __init__(
        self, 
        retriever_router: Optional[Any] = None,
        retriever_civil: Optional[Any] = None,
        retriever_penal: Optional[Any] = None,
        retriever_trabalhista: Optional[Any] = None,
        retriever_tributario: Optional[Any] = None,
        retriever_empresarial: Optional[Any] = None,
        session_id: Optional[str] = None
    ):
        self.name = "Router"
        self.model = "gpt-4o-mini"
        self.instructions = ROUTER_HANDOFF_INSTRUCTIONS
        self.session_id = session_id
        
        # Retriever do Router (banco geral)
        self._retriever = retriever_router
        self.tools = []
        if self._retriever is not None:
            buscar_documentos = criar_tool_buscar_documentos(self._retriever)
            self.tools = [buscar_documentos]
        
        # Retrievers dos especialistas
        self._retriever_civil = retriever_civil
        self._retriever_penal = retriever_penal
        self._retriever_trabalhista = retriever_trabalhista
        self._retriever_tributario = retriever_tributario
        self._retriever_empresarial = retriever_empresarial

    async def on_handoff(self, ctx: RunContextWrapper[None], input_data: SpecialistHandoffData):
        # Baseado no exemplo: quando usamos `input_type`, o callback recebe (ctx, input_data).
        # Aqui usamos para persistir o agente atual na memória da sessão.
        agent_name = getattr(getattr(ctx, "handoff", None), "agent", None)
        agent_name = getattr(agent_name, "name", None) or "unknown_agent"
        set_active_agent(session_id=self.session_id, agent_name=agent_name)
    
    def create_agent(self) -> Agent:
        
        # Cria os agentes especialistas com seus respectivos retrievers (quando disponíveis)
        handoffs_list = []
        if self._retriever_civil is not None:
            civil = CivilAgent(self._retriever_civil).create_agent()
            handoffs_list.append(handoff(agent=civil, on_handoff=self.on_handoff, input_type=SpecialistHandoffData))
        if self._retriever_penal is not None:
            penal = PenalAgent(self._retriever_penal).create_agent()
            handoffs_list.append(handoff(agent=penal, on_handoff=self.on_handoff, input_type=SpecialistHandoffData))
        if self._retriever_trabalhista is not None:
            trabalhista = TrabalhistaAgent(self._retriever_trabalhista).create_agent()
            handoffs_list.append(handoff(agent=trabalhista, on_handoff=self.on_handoff, input_type=SpecialistHandoffData))
        if self._retriever_tributario is not None:
            tributario = TributarioAgent(self._retriever_tributario).create_agent()
            handoffs_list.append(handoff(agent=tributario, on_handoff=self.on_handoff, input_type=SpecialistHandoffData))
        if self._retriever_empresarial is not None:
            empresarial = EmpresarialAgent(self._retriever_empresarial).create_agent()
            handoffs_list.append(handoff(agent=empresarial, on_handoff=self.on_handoff, input_type=SpecialistHandoffData))

        return Agent(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
            tools=self.tools,
            handoffs=handoffs_list,
        )
    
    async def run(self, query: str = None, history: Optional[List[Dict[str, Any]]] = None):
        
        history_list = history or []

        triage_agent = self.create_agent()
        
        if query:
            history_list.append({"role": "user", "content": query})
        
        result = await Runner.run(
            triage_agent,
            history_list,
            max_turns=12,
        )

        # Quando ocorre handoff, o `final_output` normalmente já é a resposta do especialista.
        return result.final_output
    
    async def run_with_trace(self, query, history=None):
        with trace("agente_router"):
            return await self.run(query, history)

    async def run_with_history(self, query, history):
       return await self.run_with_trace(query, history)

async def run_demo_loop(forcar_recriacao: bool = False):
    """
    Loop de demonstração interativo.
    
    Cria todos os retrievers necessários antes de instanciar os agentes,
    permitindo controle total sobre a inicialização dos bancos de documentos.
    
    Os vectorstores são persistidos no disco. Em execuções futuras, serão
    carregados instantaneamente ao invés de serem recriados do zero.
    
    Args:
        forcar_recriacao: Se True, recria todos os vectorstores mesmo que
                          já existam no disco (útil após atualizar documentos)
    """
    from app.agents.tools.retriever import (
        criarRetrieverBancoGeral,
        criarRetrieverCivil,
        criarRetrieverPenal,
        criarRetrieverTrabalhista,
        criarRetrieverTributario,
        criarRetrieverEmpresarial,
        listar_vectorstores_persistidos,
        VECTORSTORE_PERSIST_DIR,
    )
    
    print("="*60)
    print("INICIALIZANDO SISTEMA DE AGENTES JURÍDICOS")
    print("="*60)
    
    # Verificar vectorstores existentes
    vectorstores_existentes = listar_vectorstores_persistidos()
    if vectorstores_existentes and not forcar_recriacao:
        print(f"\n[INFO] Vectorstores persistidos encontrados em: {VECTORSTORE_PERSIST_DIR}")
        print(f"       Coleções: {', '.join(vectorstores_existentes)}")
        print("       Os vectorstores serão carregados do disco (muito mais rápido!)")
    elif forcar_recriacao:
        print("\n[INFO] Modo FORÇAR RECRIAÇÃO ativado.")
        print("       Todos os vectorstores serão recriados do zero.")
    else:
        print("\n[INFO] Nenhum vectorstore persistido encontrado.")
        print("       Os vectorstores serão criados e salvos em disco.")
        print("       (Isso pode demorar na primeira execução)")
    
    print("\n" + "-"*60)
    
    # 1. Criar/carregar todos os retrievers
    print("\n[1/6] Retriever do BANCO GERAL (Router)...")
    retriever_router = criarRetrieverBancoGeral(forcar_recriacao=forcar_recriacao)
    
    print("\n[2/6] Retriever CIVIL...")
    retriever_civil = criarRetrieverCivil()
    
    print("\n[3/6] Retriever PENAL...")
    retriever_penal = criarRetrieverPenal()
    
    print("\n[4/6] Retriever TRABALHISTA...")
    retriever_trabalhista = criarRetrieverTrabalhista()
    
    print("\n[5/6] Retriever TRIBUTÁRIO...")
    retriever_tributario = criarRetrieverTributario()
    
    print("\n[6/6] Retriever EMPRESARIAL...")
    retriever_empresarial = criarRetrieverEmpresarial()
    
    print("\n" + "="*60)
    print("TODOS OS RETRIEVERS PRONTOS!")
    print("="*60)
    
    # 2. Criar sessão
    session_id = os.getenv("SESSION_ID") or str(uuid.uuid4())
    
    # 3. Instanciar o Router com todos os retrievers
    router = Router(
        retriever_router=retriever_router,
        retriever_civil=retriever_civil,
        retriever_penal=retriever_penal,
        retriever_trabalhista=retriever_trabalhista,
        retriever_tributario=retriever_tributario,
        retriever_empresarial=retriever_empresarial,
        session_id=session_id
    )
    
    session_state = get_session_state(session_id=session_id)
    history = cast(List[Dict[str, Any]], session_state["historico"])

    print("\n" + "="*60)
    print("SISTEMA PRONTO! Digite sua pergunta jurídica.")
    print("Comandos especiais:")
    print("  'sair'     - Encerrar o sistema")
    print("  'limpar'   - Limpar histórico da conversa")
    print("="*60 + "\n")

    while True:

        query = input("Você: ").strip()
        
        if query.lower() == "sair":
            print("\nEncerrando sistema...")
            break
        
        if query.lower() == "limpar":
            history.clear()
            save_memory(session_id=session_id, memory=history)
            print("\n[INFO] Histórico limpo!\n")
            continue
        
        if not query:
            continue

        result = await router.run_with_history(query=query, history=history)
        print('\n' + '-'*40)
        print(f"Assistente:\n{result}")
        print('-'*40 + '\n')
        
        history.append({"role": "assistant", "content": result})
        save_memory(session_id=session_id, memory=history)

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema de Agentes Jurídicos com RAG")
    parser.add_argument(
        "-r", "--rebuild",
        action="store_true",
        help="Força a recriação de todos os vectorstores (ignora cache do disco)"
    )
    parser.add_argument(
        "--limpar-vectorstores",
        action="store_true", 
        help="Remove todos os vectorstores persistidos e encerra"
    )
    
    args = parser.parse_args()
    
    if args.limpar_vectorstores:
        from app.agents.tools.retriever import limpar_todos_vectorstores
        print("Removendo todos os vectorstores persistidos...")
        count = limpar_todos_vectorstores()
        print(f"Removidos {count} vectorstores.")
    else:
        asyncio.run(run_demo_loop(forcar_recriacao=args.rebuild))