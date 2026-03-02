import uuid
from typing import Any, Dict, List, cast

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_retriever
from app.agents.assistants.civil import CivilAgent
from app.agents.assistants.empresarial import EmpresarialAgent
from app.agents.assistants.penal import PenalAgent
from app.agents.assistants.router import Router
from app.agents.assistants.trabalhista import TrabalhistaAgent
from app.agents.assistants.tributario import TributarioAgent
from app.agents.tools.memory import get_active_agent, get_session_state, save_memory
from app.models.inputs import QuestionInput
from app.models.outputs import AgentResponse

router = APIRouter()

AGENT_MAP = {
    "Civil": CivilAgent,
    "Penal": PenalAgent,
    "Trabalhista": TrabalhistaAgent,
    "Tributario": TributarioAgent,
    "Empresarial": EmpresarialAgent,
}

AGENT_NAME_MAP = {
    "Especialista_Civil": "Civil",
    "Especialista_Penal": "Penal",
    "Especialista_Trabalhista": "Trabalhista",
    "Especialista_Tributario": "Tributario",
    "Especialista_Empresarial": "Empresarial",
}


@router.post("/question", response_model=AgentResponse)
async def process_question(
    input_data: QuestionInput,
    retriever=Depends(get_retriever),
) -> AgentResponse:
    """
    Processa uma pergunta jurídica e retorna a resposta do agente.

    - Se `agente_selecionado` for especificado, usa esse agente diretamente
    - Caso contrário, o Router decide qual agente usar
    """
    try:
        session_id = input_data.session_id or str(uuid.uuid4())

        session_state = get_session_state(session_id)
        history = cast(List[Dict[str, Any]], session_state["historico"])

        if input_data.agente_selecionado:
            agent_class = AGENT_MAP.get(input_data.agente_selecionado)
            if not agent_class:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agente inválido: {input_data.agente_selecionado}",
                )
            agent = agent_class(retriever)
            agente_selecionado = input_data.agente_selecionado
        else:
            agent = Router(retriever=retriever, session_id=session_id)
            agente_selecionado = "Router"

        result = await agent.run_with_history(
            query=input_data.pergunta,
            history=history.copy(),
        )

        history.append({"role": "user", "content": input_data.pergunta})
        history.append({"role": "assistant", "content": result})
        save_memory(session_id=session_id, memory=history)

        if agente_selecionado == "Router":
            agente_raw = get_active_agent(session_id=session_id)
            agente_respondendo = AGENT_NAME_MAP.get(agente_raw, "Civil")
        else:
            agente_respondendo = agente_selecionado

        return AgentResponse(
            agente_selecionado=agente_selecionado,
            agente_respondendo=agente_respondendo,
            resposta=str(result),
            referencias=[],
            session_id=session_id,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar pergunta: {str(exc)}",
        )
