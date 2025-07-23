from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import shap
from lightgbm import LGBMClassifier

app = FastAPI()

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "LGBMClassifier1.pkl")
test_path = os.path.join(current_dir, "application_test_subset.csv")

# Chargement modèle
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Chargement données avec ou sans TARGET selon besoin
def load_test_data(drop_target=True):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Fichier introuvable : {test_path}")
    df = pd.read_csv(test_path, index_col=0)
    if drop_target and "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    return df

def load_test_data_with_target():
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Fichier introuvable : {test_path}")
    df = pd.read_csv(test_path, index_col=0)
    # on garde TARGET
    return df

class ClientID(BaseModel):
    client_id: int

class PredictionResult(BaseModel):
    prediction: int
    probability: float

@app.get("/")
def welcome():
    return {"message": "API de prédiction de crédit"}

@app.get("/clients")
def get_client_ids():
    df = load_test_data()
    return {"client_ids": df.index.tolist()}

@app.get("/all_data")
def get_all_data():
    # On renvoie tout avec TARGET si elle existe
    df = load_test_data_with_target()
    df_reset = df.reset_index()
    return df_reset.to_dict(orient="records")

@app.get("/mean_class_1")
def mean_of_class_1():
    df = load_test_data()
    probas = model.predict_proba(df)[:, 1]
    df["proba_defaut"] = probas
    class_1_df = df[df["proba_defaut"] >= 0.2]
    if "proba_defaut" in class_1_df.columns:
        class_1_df = class_1_df.drop(columns=["proba_defaut"])
    return class_1_df.mean().to_dict()

@app.post("/predict", response_model=PredictionResult)
def predict(client: ClientID):
    df = load_test_data()
    if client.client_id not in df.index:
        raise HTTPException(status_code=404, detail="Client ID non trouvé")
    client_input = pd.DataFrame([df.loc[client.client_id]])
    proba = float(model.predict_proba(client_input)[0][1])
    pred = int(proba >= 0.2)
    return PredictionResult(prediction=pred, probability=proba)

@app.post("/client_data")
def get_client_data(client: ClientID):
    df = load_test_data()
    
    if client.client_id not in df.index:
        raise HTTPException(status_code=404, detail="Client non trouvé.")
    return df.loc[client.client_id].to_dict()

@app.post("/explain")
def explain(client: ClientID):
    df = load_test_data()
    if client.client_id not in df.index:
        raise HTTPException(status_code=404, detail="Client ID non trouvé")
    client_input = pd.DataFrame([df.loc[client.client_id]])
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(client_input)
    if isinstance(shap_values_all, list):
        shap_values = shap_values_all[1]
    else:
        shap_values = shap_values_all
    shap_df = pd.Series(shap_values[0], index=client_input.columns).abs()
    top5 = shap_df.sort_values(ascending=False).head(5)
    return top5.to_dict()

@app.post("/explain_full")
def explain_full(client: ClientID):
    df = load_test_data()
    if client.client_id not in df.index:
        raise HTTPException(status_code=404, detail="Client ID non trouvé")
    client_input = pd.DataFrame([df.loc[client.client_id]])
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(client_input)
    if isinstance(shap_values_all, list):
        shap_values = shap_values_all[1]
    else:
        shap_values = shap_values_all
    shap_series = pd.Series(shap_values[0], index=client_input.columns)
    return shap_series.to_dict()

@app.post("/compare_client_group_class_1")
def compare_client_group_class_1(client: ClientID):
    df = load_test_data()
    probas = model.predict_proba(df)[:, 1]
    df["proba_defaut"] = probas

    if client.client_id not in df.index:
        raise HTTPException(status_code=404, detail="Client ID non trouvé")

    client_data = df.loc[client.client_id]
    class_1_df = df[df["proba_defaut"] >= 0.2]

    # Supprimer 'proba_defaut' seulement si présent
    if "proba_defaut" in class_1_df.columns:
        mean_class_1 = class_1_df.drop(columns=["proba_defaut"]).mean()
    else:
        mean_class_1 = class_1_df.mean()

    if "proba_defaut" in client_data.index:
        client_data = client_data.drop("proba_defaut")

    comparison = client_data - mean_class_1
    return comparison.to_dict()
