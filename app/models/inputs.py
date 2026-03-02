"""
Contratos de entrada (incoming) da API para interação com agentes jurídicos
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class QuestionInput(BaseModel):
    """
    Modelo de entrada (incoming) para perguntas do usuário
    """
    agente_selecionado: Optional[Literal["Civil", "Penal", "Trabalhista", "Tributario", "Empresarial"]] = Field(
        None,
        description="Agente específico selecionado pelo usuário. Se None, o Router decide automaticamente"
    )
    pergunta: str = Field(
        ...,
        description="Pergunta ou questão jurídica do usuário",
        min_length=1
    )
    midias: List[str] = Field(
        default_factory=list,
        description="Lista de URLs de mídias anexadas no S3 (documentos, imagens, etc.)"
    )
    session_id: Optional[str] = Field(
        None,
        description="ID da sessão para manter histórico de conversação"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "agente_selecionado": "Trabalhista",
                "pergunta": "Quais são meus direitos em caso de demissão sem justa causa?",
                "midias": [
                    "https://s3.amazonaws.com/bucket/contrato_trabalho.pdf",
                    "https://s3.amazonaws.com/bucket/documento.pdf"
                ],
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
