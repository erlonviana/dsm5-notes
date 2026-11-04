# src/treinar_pipeline_churn.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def obter_dataset() -> pd.DataFrame:
    """Carrega e padroniza o arquivo CSV do dataset Telco Customer Churn (Kaggle)."""
    caminho_csv = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Arquivo '{caminho_csv}' não encontrado na raiz do projeto. "
            "Baixe o dataset Telco Customer Churn no Kaggle e salve-o na pasta raiz."
        )

    print(f"Carregando dataset do Kaggle: {caminho_csv}")
    df_raw = pd.read_csv(caminho_csv)

    df = pd.DataFrame()
    df["tempo_cliente_meses"] = pd.to_numeric(df_raw["tenure"], errors="coerce").fillna(0).astype(int)
    df["valor_mensal"] = pd.to_numeric(df_raw["MonthlyCharges"], errors="coerce").fillna(0.0)

    mapa_contrato = {"Month-to-month": "mensal", "One year": "anual", "Two year": "bienal"}
    df["tipo_contrato"] = df_raw["Contract"].map(mapa_contrato).fillna("mensal")

    def mapa_pgto(v):
        txt = str(v).lower()
        return "boleto" if ("electronic" in txt or "mailed" in txt) else ("cartao" if "credit" in txt else "pix")

    df["forma_pagamento"] = df_raw["PaymentMethod"].apply(mapa_pgto)
    df["churn"] = (df_raw["Churn"] == "Yes").astype(int)
    return df


def main():
    df = obter_dataset()
    X = df.drop(columns=["churn"])
    y = df["churn"]

    colunas_num = ["tempo_cliente_meses", "valor_mensal"]
    colunas_cat = ["tipo_contrato", "forma_pagamento"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1. Transformador unificado com StandardScaler e OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colunas_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_cat),
        ]
    )

    # 2. Pipeline unindo Pré-processamento e Rede Neural MLP
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classificador",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Treinando pipeline com rede neural MLP...")
    pipeline.fit(X_train, y_train)

    predicoes = pipeline.predict(X_test)
    print(f"Acurácia de teste: {accuracy_score(y_test, predicoes) * 100:.2f}%")
    print(classification_report(y_test, predicoes))

    # 3. Exportação do objeto completo
    Path("models").mkdir(exist_ok=True)
    caminho_saida = "models/pipeline_churn.joblib"
    joblib.dump(pipeline, caminho_saida)
    print(f"Pipeline serializado com sucesso em: {caminho_saida}")


if __name__ == "__main__":
    main()