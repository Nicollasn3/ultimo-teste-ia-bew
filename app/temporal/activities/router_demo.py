from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, cast
from temporalio import activity
from app.agents.assistants.router import Router
from app.agents.tools.memory import get_session_state, save_memory
from app.agents.tools.retriever import (
    VECTORSTORE_PERSIST_DIR,
    criarRetrieverBancoGeral,
    criarRetrieverCivil,
    criarRetrieverEmpresarial,
    criarRetrieverPenal,
    criarRetrieverTrabalhista,
    criarRetrieverTributario,
    listar_vectorstores_persistidos,
)

_RETRIEVERS_CACHE: Dict[
    Tuple[bool, bool, bool, bool, bool, bool],
    Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]],
] = {}

def _log_vectorstore_status(forcar_recriacao: bool) -> None:
    vectorstores_existentes = listar_vectorstores_persistidos()
    if vectorstores_existentes and not forcar_recriacao:
        activity.logger.info(
            "Vectorstores persistidos encontrados em %s (colecoes: %s).",
            VECTORSTORE_PERSIST_DIR,
            ", ".join(vectorstores_existentes),
        )
    elif forcar_recriacao:
        activity.logger.info(
            "Modo FORCAR RECRIACAO ativado. Vectorstores serao recriados."
        )
    else:
        activity.logger.info(
            "Nenhum vectorstore persistido encontrado. Serao criados e salvos."
        )

def _get_or_create_retrievers(
    forcar_recriacao: bool,
    use_router: bool = True,
    use_civil: bool = True,
    use_penal: bool = True,
    use_trabalhista: bool = True,
    use_tributario: bool = True,
    use_empresarial: bool = True,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    cache_key = (
        use_router,
        use_civil,
        use_penal,
        use_trabalhista,
        use_tributario,
        use_empresarial,
    )

    if not forcar_recriacao and cache_key in _RETRIEVERS_CACHE:
        return _RETRIEVERS_CACHE[cache_key]

    _log_vectorstore_status(forcar_recriacao)
    retriever_router = criarRetrieverBancoGeral(forcar_recriacao=forcar_recriacao) if use_router else None
    retriever_civil = criarRetrieverCivil() if use_civil else None
    retriever_penal = criarRetrieverPenal() if use_penal else None
    retriever_trabalhista = criarRetrieverTrabalhista() if use_trabalhista else None
    retriever_tributario = criarRetrieverTributario() if use_tributario else None
    retriever_empresarial = criarRetrieverEmpresarial() if use_empresarial else None
    _RETRIEVERS_CACHE[cache_key] = (
        retriever_router,
        retriever_civil,
        retriever_penal,
        retriever_trabalhista,
        retriever_tributario,
        retriever_empresarial,
    )

    return _RETRIEVERS_CACHE[cache_key]

@activity.defn
async def run_demo_loop_activity(
    query: str,
    session_id: str,
    forcar_recriacao: bool = False,
    clear_history: bool = False,
    use_router_retriever: bool = False,
    use_civil_retriever: bool = False,
    use_penal_retriever: bool = False,
    use_trabalhista_retriever: bool = False,
    use_tributario_retriever: bool = False,
    use_empresarial_retriever: bool = False,
) -> str:
    
    if clear_history:
        session_state = get_session_state(session_id=session_id)
        history = cast(List[Dict[str, Any]], session_state["historico"])
        history.clear()
        save_memory(session_id=session_id, memory=history)
        return "[INFO] Historico limpo!"

    if not query:
        return ""

    (
        retriever_router,
        retriever_civil,
        retriever_penal,
        retriever_trabalhista,
        retriever_tributario,
        retriever_empresarial,
    ) = _get_or_create_retrievers(
        forcar_recriacao=forcar_recriacao,
        use_router=use_router_retriever,
        use_civil=use_civil_retriever,
        use_penal=use_penal_retriever,
        use_trabalhista=use_trabalhista_retriever,
        use_tributario=use_tributario_retriever,
        use_empresarial=use_empresarial_retriever,
    )

    router = Router(
        retriever_router=retriever_router,
        retriever_civil=retriever_civil,
        retriever_penal=retriever_penal,
        retriever_trabalhista=retriever_trabalhista,
        retriever_tributario=retriever_tributario,
        retriever_empresarial=retriever_empresarial,
        session_id=session_id,
    )

    session_state = get_session_state(session_id=session_id)
    history = cast(List[Dict[str, Any]], session_state["historico"])

    result = await router.run_with_history(query=query, history=history)
    history.append({"role": "assistant", "content": result})
    save_memory(session_id=session_id, memory=history)
    return result
