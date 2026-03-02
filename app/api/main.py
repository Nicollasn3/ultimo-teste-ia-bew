from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.tools.retriever import criarRetriever
from app.api.routes import question


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa recursos caros uma única vez."""
    print("Inicializando retriever...")
    app.state.retriever = criarRetriever()
    print("Retriever pronto!")
    yield
    print("Encerrando servidor...")


app = FastAPI(
    title="BEW Advogados - IA Service",
    description="API para agentes jurídicos especializados",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(question.router, prefix="/api", tags=["Question"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ia-service"}
