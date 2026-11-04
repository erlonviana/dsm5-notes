# # src/app.py
# from contextlib import asynccontextmanager
# from pathlib import Path
# import joblib
# import numpy as np
# from fastapi import FastAPI, HTTPException, status

# from src.schemas import PontoInput, PrevisaoOutput

# # Dicionário global que mantém o modelo residente na memória RAM
# recursos: dict = {}


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Carrega o modelo do disco para a RAM uma única vez no startup."""
#     caminho_modelo = Path("models/mlp_moons.joblib")

#     if not caminho_modelo.exists():
#         raise RuntimeError(
#             f"Arquivo de modelo não encontrado em: {caminho_modelo}. "
#             "Execute a prática anterior para exportar o arquivo."
#         )

#     recursos["modelo"] = joblib.load(caminho_modelo)
#     print(f"[STARTUP] Modelo carregado na memória: {caminho_modelo}")

#     yield

#     recursos.clear()
#     print("[SHUTDOWN] Memória RAM liberada.")


# app = FastAPI(
#     title="API de Predição com Redes Neurais",
#     description="Inferência em tempo real com Scikit-Learn e FastAPI",
#     version="1.0.0",
#     lifespan=lifespan,
# )


# @app.get("/health", tags=["Status"])
# def verificar_saude():
#     """Verifica a integridade da API e a disponibilidade do modelo."""
#     return {
#         "status": "online",
#         "modelo_carregado": "modelo" in recursos,
#     }


# @app.post("/predict", response_model=PrevisaoOutput, tags=["Predição"])
# def prever_ponto(ponto: PontoInput):
#     """Executa a inferência para um ponto cartesiano bidimensional."""
#     modelo = recursos.get("modelo")
#     if modelo is None:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Modelo preditivo não disponível em memória.",
#         )

#     # Converte o schema validado em matriz 2D para o Scikit-Learn
#     dados = np.array([[ponto.x1, ponto.x2]])

#     classe = int(modelo.predict(dados)[0])
#     probabilidades = modelo.predict_proba(dados)[0]
#     prob_classe_1 = float(probabilidades[1])

#     rotulo = "Classe Superior (1)" if classe == 1 else "Classe Inferior (0)"

#     return PrevisaoOutput(
#         classe=classe,
#         rotulo=rotulo,
#         probabilidade=round(prob_classe_1, 4),
#     )

# src/app.py
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.schemas import (
    PontoInput,
    PrevisaoOutput,
    ClienteChurnInput,
    ClienteChurnOutput,
)

# Dicionário global que mantém os modelos residentes na memória RAM
recursos: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os modelos do disco para a RAM uma única vez no startup."""
    caminho_modelo = Path("models/mlp_moons.joblib")
    if caminho_modelo.exists():
        recursos["modelo"] = joblib.load(caminho_modelo)
        print(f"[STARTUP] Modelo carregado na memória: {caminho_modelo}")

    caminho_churn = Path("models/pipeline_churn.joblib")
    if caminho_churn.exists():
        recursos["pipeline_churn"] = joblib.load(caminho_churn)
        print(f"[STARTUP] Pipeline de Churn carregado na memória: {caminho_churn}")

    yield

    recursos.clear()
    print("[SHUTDOWN] Memória RAM liberada.")


app = FastAPI(
    title="API de Predição com Redes Neurais",
    description="Inferência em tempo real com Scikit-Learn e FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Status"])
def verificar_saude():
    """Verifica a integridade da API e a disponibilidade dos modelos."""
    return {
        "status": "online",
        "modelo_carregado": "modelo" in recursos,
        "pipeline_churn_carregado": "pipeline_churn" in recursos,
    }


@app.post("/predict", response_model=PrevisaoOutput, tags=["Predição"])
def prever_ponto(ponto: PontoInput):
    """Executa a inferência para um ponto cartesiano bidimensional."""
    modelo = recursos.get("modelo")
    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo preditivo não disponível em memória.",
        )

    dados = np.array([[ponto.x1, ponto.x2]])

    classe = int(modelo.predict(dados)[0])
    probabilidades = modelo.predict_proba(dados)[0]
    prob_classe_1 = float(probabilidades[1])

    rotulo = "Classe Superior (1)" if classe == 1 else "Classe Inferior (0)"

    return PrevisaoOutput(
        classe=classe,
        rotulo=rotulo,
        probabilidade=round(prob_classe_1, 4),
    )


@app.post("/predict/churn", response_model=ClienteChurnOutput, tags=["Produção: Churn Kaggle"])
def prever_churn(cliente: ClienteChurnInput):
    """Executa inferência com o pipeline completo treinado com dados do Kaggle."""
    pipeline = recursos.get("pipeline_churn")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline de Churn não disponível na memória.",
        )

    # Converte os dados brutos validados diretamente em DataFrame
    df_cliente = pd.DataFrame([cliente.model_dump()])

    predicao = int(pipeline.predict(df_cliente)[0])
    probabilidade = float(pipeline.predict_proba(df_cliente)[0][1])

    if predicao == 1:
        rotulo = "Alto Risco de Cancelamento"
        recomendacao = "Oferecer desconto promocional imediato ou migração de plano."
    else:
        rotulo = "Baixo Risco (Cliente Fiel)"
        recomendacao = "Manter rotina padrão de atendimento e pós-venda."

    return ClienteChurnOutput(
        previsao_churn=predicao,
        rotulo=rotulo,
        probabilidade_churn=round(probabilidade, 4),
        recomendacao=recomendacao,
    )