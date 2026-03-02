from __future__ import annotations
import asyncio
import os
from temporalio.client import Client
from temporalio.worker import Worker
from app.temporal.activities.router_demo import run_demo_loop_activity
from app.temporal.workflows.conversational_chat import ConversationalChatWorkflow

async def main() -> None:

    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "conversational-chat")

    client = await Client.connect(temporal_host)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[ConversationalChatWorkflow],
        activities=[run_demo_loop_activity],
    )
    
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
