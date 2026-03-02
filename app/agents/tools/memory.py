from __future__ import annotations
from pathlib import Path
from typing import Dict, List, TypedDict
import json
from json import JSONDecodeError

# Define o diretório de memória na raiz do projeto
# Pega o diretório do arquivo atual e sobe até a raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

# Debug desabilitado para evitar prints duplicados durante imports
# print(f"Diretório de memória: {MEMORY_DIR.absolute()}")

DEFAULT_AGENT = "triagem"

class SessionState(TypedDict):
    agente_atual: str
    historico: List[Dict]

def _get_session_file(session_id: str) -> Path:
    return MEMORY_DIR / f"{session_id}.json"

def _load_raw_state(session_id: str) -> SessionState:
    file_path = _get_session_file(session_id)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return SessionState(agente_atual=DEFAULT_AGENT, historico=data)
                if isinstance(data, dict):
                    historico = data.get("historico", [])
                    agente_atual = data.get("agente_atual", DEFAULT_AGENT)
                    return SessionState(agente_atual=agente_atual, historico=historico)
        except JSONDecodeError:
            file_path.unlink(missing_ok=True)
    return SessionState(agente_atual=DEFAULT_AGENT, historico=[])

def _save_raw_state(session_id: str, state: SessionState):
    file_path = _get_session_file(session_id)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def load_memory(session_id: str) -> List[Dict]:
    """Retorna apenas o histórico da sessão."""
    return _load_raw_state(session_id)["historico"]

def save_memory(session_id: str, memory: List[Dict]):
    """Atualiza o histórico mantendo o agente atual."""
    state = _load_raw_state(session_id)
    state["historico"] = memory
    _save_raw_state(session_id, state)

def clear_memory(session_id: str):
    """Remove completamente o arquivo da sessão."""
    file_path = _get_session_file(session_id)
    print(f"🗑️  Tentando limpar memória: {file_path.absolute()}")
    if file_path.exists():
        file_path.unlink()
        print(f"✅ Memória limpa com sucesso. Session ID: {session_id}")
    else:
        print(f"⚠️  Arquivo de memória não existe: {file_path.absolute()}")

def get_session_state(session_id: str) -> SessionState:
    return _load_raw_state(session_id)

def set_session_state(session_id: str, state: SessionState):
    _save_raw_state(session_id, state)

def get_active_agent(session_id: str) -> str:
    return _load_raw_state(session_id)["agente_atual"]

def set_active_agent(session_id: str, agent_name: str):
    state = _load_raw_state(session_id)
    state["agente_atual"] = agent_name
    _save_raw_state(session_id, state)