from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from agents import Agent

class BaseAgent(ABC):
    @abstractmethod
    def create_agent(self) -> Agent:
        return Agent(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
        )

    @abstractmethod
    async def run(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> Any:
        pass

    @abstractmethod
    async def run_with_trace(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> Any:
        pass


