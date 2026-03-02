from contextlib import contextmanager
from typing import Iterator


@contextmanager
def trace(name: str) -> Iterator[None]:
    """
    Trace local (fallback) para instrumentação simples via `with trace("...")`.
    Mantém compatibilidade com o padrão usado pelos agentes do projeto.
    """
    yield

