# from pydantic import BaseModel, Field

# class PontoInput(BaseModel):
#     x1: float = Field(..., description="Coordenada X1 do ponto", examples=[0.5])
#     x2: float = Field(..., description="Coordenada X1 do ponto", examples=[0.2])

# class PrevisaoOutput(BaseModel):
#     classe: int
#     rotulo: str
#     probabilidade: float

# src/schemas.py
from typing import Literal
from pydantic import BaseModel, Field


class PontoInput(BaseModel):
    """Estrutura dos dados de entrada para predição cartesiana."""
    x1: float = Field(..., description="Coordenada X1 do ponto", examples=[0.5])
    x2: float = Field(..., description="Coordenada X2 do ponto", examples=[0.2])


class PrevisaoOutput(BaseModel):
    """Estrutura do resultado devolvido pela API."""
    classe: int
    rotulo: str
    probabilidade: float


class ClienteChurnInput(BaseModel):
    """Dados brutos de cliente para análise de risco de cancelamento."""
    tempo_cliente_meses: int = Field(..., ge=0, le=120, description="Meses de contrato ativos", examples=[3])
    valor_mensal: float = Field(..., ge=0.0, description="Mensalidade cobrada em reais", examples=[149.90])
    tipo_contrato: Literal["mensal", "anual", "bienal"] = Field(..., description="Periodicidade contratual", examples=["mensal"])
    forma_pagamento: Literal["cartao", "boleto", "pix"] = Field(..., description="Forma de pagamento", examples=["boleto"])


class ClienteChurnOutput(BaseModel):
    """Diagnóstico preditivo de cancelamento."""
    previsao_churn: int
    rotulo: str
    probabilidade_churn: float
    recomendacao: str