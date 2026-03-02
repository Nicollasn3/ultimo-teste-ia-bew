from __future__ import annotations

from datetime import timedelta
from typing import Optional
from temporalio import workflow
with workflow.unsafe.imports_passed_through():
    from app.temporal.activities.router_demo import run_demo_loop_activity

@workflow.defn
class ConversationalChatWorkflow:
    
    def __init__(self) -> None:
        self._session_id: Optional[str] = None
        self._forcar_recriacao = False

    @workflow.run
    async def run(self, session_id: str, forcar_recriacao: bool = False) -> None:
        self._session_id = session_id
        self._forcar_recriacao = forcar_recriacao
        await workflow.wait_condition(lambda: False)

    @workflow.update(name="send_message")
    async def send_message(self, query: str, clear_history: bool = False) -> str:
        if not self._session_id:
            raise RuntimeError("Workflow nao inicializado.")
        return await workflow.execute_activity(
            run_demo_loop_activity,
            args=[query, self._session_id, self._forcar_recriacao, clear_history],
            start_to_close_timeout=timedelta(minutes=10),
        )
