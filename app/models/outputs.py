"""
Contratos de saída (outgoing) da API para respostas dos agentes jurídicos
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Reference(BaseModel):
    """
    Modelo para referências jurídicas utilizadas na resposta
    """
    titulo: str = Field(
        ...,
        description="Título ou identificação da referência",
        examples=["Art. 477 da CLT", "Súmula 331 do TST"]
    )
    fonte: str = Field(
        ...,
        description="Fonte da referência (CPC, CLT, Lei específica, STF, STJ, etc.)",
        examples=["CLT", "CPC", "Lei 8.078/90 (CDC)", "STF", "STJ"]
    )
    conteudo: Optional[str] = Field(
        None,
        description="Trecho relevante do texto legal ou jurisprudencial"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadados adicionais (URL, data, número do processo, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "titulo": "Art. 477 da CLT",
                "fonte": "CLT - Consolidação das Leis do Trabalho",
                "conteudo": "É assegurado a todo empregado, não existindo prazo estipulado para a terminação do respectivo contrato...",
                "metadata": {
                    "url": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm",
                    "artigo": "477",
                    "lei": "Decreto-Lei 5.452/1943"
                }
            }
        }


class AgentResponse(BaseModel):
    """
    Modelo de saída (outgoing) com a resposta do agente
    """
    agente_selecionado: Literal["Router", "Civil", "Penal", "Trabalhista", "Tributario", "Empresarial"] = Field(
        ...,
        description="Agente que foi selecionado/solicitado inicialmente (pode ser 'Router' se automático)"
    )
    agente_respondendo: Literal["Civil", "Penal", "Trabalhista", "Tributario", "Empresarial"] = Field(
        ...,
        description="Agente que está efetivamente respondendo a pergunta"
    )
    resposta: str = Field(
        ...,
        description="Resposta textual do agente à pergunta do usuário",
        min_length=1
    )
    referencias: List[Reference] = Field(
        default_factory=list,
        description="Lista de referências jurídicas utilizadas na resposta"
    )
    session_id: str = Field(
        ...,
        description="ID da sessão da conversação"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "agente_selecionado": "Trabalhista",
                "agente_respondendo": "Trabalhista",
                "resposta": "Em caso de demissão sem justa causa, você tem direito a: aviso prévio (trabalhado ou indenizado), saldo de salário, férias vencidas e proporcionais com 1/3, 13º salário proporcional, saque do FGTS com multa de 40%, e seguro-desemprego (se cumprir os requisitos).",
                "referencias": [
                    {
                        "titulo": "Art. 477 da CLT",
                        "fonte": "CLT - Consolidação das Leis do Trabalho",
                        "conteudo": "É assegurado a todo empregado, não existindo prazo estipulado para a terminação do respectivo contrato...",
                        "url": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm"
                    },
                    {
                        "titulo": "Art. 18 da Lei 8.036/90",
                        "fonte": "Lei do FGTS",
                        "conteudo": "O saldo da conta vinculada será pago quando ocorrer despedida sem justa causa...",
                        "url": "https://www.planalto.gov.br/ccivil_03/leis/l8036compilada.htm"
                    }
                ],
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
