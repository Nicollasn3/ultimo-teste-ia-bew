import os
import sys
from typing import Any, Dict, List, Optional

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from agents import Agent, Runner
from app.agents.prompts.prompts import SPECIALIST_PROMPTS_BY_AREA
from app.agents.shared.base_agent import BaseAgent
from app.agents.utils.trace import trace
from app.agents.tools.retriever import criar_tool_buscar_documentos

class TributarioAgent(BaseAgent):
    """
    Agente especialista em Direito Tributário.
    
    Base documental específica:
    - Direito Tributário (36 docs)
    - Direito Financeiro Brasileiro (3 docs)
    + Banco Geral (Constitucional, Processo Civil, Códigos, etc.)
    
    Args:
        retriever: Retriever configurado para busca de documentos (opcional)
    """
    
    def __init__(self, retriever: Optional[Any] = None):
        self.name = "Especialista_Tributario"
        self.model = "gpt-4o-mini"
        self.instructions = SPECIALIST_PROMPTS_BY_AREA["tributario"]
        self._retriever = retriever
        self.tools = []
        if self._retriever is not None:
            buscar_documentos = criar_tool_buscar_documentos(self._retriever)
            self.tools = [buscar_documentos]

    def create_agent(self) -> Agent:
        return Agent(
            name=self.name,
            handoff_description="Especialista em Direito Tributário (Brasil) - Tributos, Lançamento, Prescrição/Decadência, Defesa Fiscal",
            instructions=self.instructions,
            model=self.model,
            tools=self.tools,
        )

    async def run(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> Any:
        history_list = history or []

        if query:
            history_list.append({"role": "user", "content": query})

        agent = self.create_agent()
        result = await Runner.run(
            agent,
            history_list,
            max_turns=12,
        )
        return result.final_output

    async def run_with_trace(self, query, history=None):
        with trace("agente_especialista_tributario"):
            return await self.run(query, history)

    async def run_with_history(self, query, history):
        return await self.run_with_trace(query, history)

