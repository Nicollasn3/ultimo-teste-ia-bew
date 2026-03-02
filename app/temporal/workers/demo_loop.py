from __future__ import annotations

import asyncio
import os
import uuid
from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode
from app.temporal.workflows.conversational_chat import ConversationalChatWorkflow

async def main() -> None:
    
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "conversational-chat")
    workflow_id = os.getenv("TEMPORAL_WORKFLOW_ID") or f"conversational-chat-{uuid.uuid4()}"
    session_id = os.getenv("SESSION_ID") or str(uuid.uuid4())
    forcar_recriacao = os.getenv("FORCAR_RECRIACAO", "").lower() in {"1", "true", "yes"}

    client = await Client.connect(temporal_host)

    try:
        handle = await client.start_workflow(
            ConversationalChatWorkflow.run,
            args=[session_id, forcar_recriacao],
            id=workflow_id,
            task_queue=task_queue,
        )
    except RPCError as exc:
        if exc.status == RPCStatusCode.ALREADY_EXISTS:
            handle = client.get_workflow_handle(workflow_id)
        else:
            raise

    print("=" * 60)
    print("SISTEMA PRONTO! Digite sua pergunta juridica.")
    print("Comandos especiais:")
    print("  'sair'     - Encerrar o sistema")
    print("  'limpar'   - Limpar historico da conversa")
    print("=" * 60 + "\n")

    while True:
        query = input("Voce: ").strip()
        if query.lower() == "sair":
            print("\nEncerrando sistema...")
            break

        if query.lower() == "limpar":
            result = await handle.execute_update("send_message", args=[query, True])
            print(f"\n{result}\n")
            continue

        if not query:
            continue

        result = await handle.execute_update("send_message", args=[query, False])
        print("\n" + "-" * 40)
        print(f"Assistente:\n{result}")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
