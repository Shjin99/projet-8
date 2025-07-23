import streamlit as st
import requests
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- Config ---
API_BASE = "http://127.0.0.1:8000"
st.set_page_config(page_title="Dashboard Crédit Client", layout="wide")
st.title("📊 Dashboard de Prédiction de Crédit Client")

# --- Chargement des IDs clients ---
with st.spinner("🔄 Chargement des clients..."):
    response = requests.get(f"{API_BASE}/clients")
    if response.status_code != 200:
        st.error("Erreur lors du chargement des IDs clients.")
        st.stop()
    client_ids = response.json()["client_ids"]

selected_client_id = st.selectbox("🧑 Sélectionner un ID client :", client_ids)

if st.button("🔍 Lancer l’analyse"):

    # --- Prédiction ---
    pred_response = requests.post(f"{API_BASE}/predict", json={"client_id": selected_client_id})
    if pred_response.status_code != 200:
        st.error("Erreur lors de la prédiction.")
        st.stop()
    pred_result = pred_response.json()
    score = round(pred_result["probability"] * 100, 2)
    decision = "❌ Défaut probable" if pred_result["prediction"] else "✅ Crédit accordé"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Probabilité de défaut", f"{score} %")
        st.metric("Décision", decision)

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Score Crédit (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if score >= 20 else "green"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 100], 'color': "lightcoral"},
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Données Globales ---
    data_response = requests.get(f"{API_BASE}/all_data")
    if data_response.status_code != 200:
        st.error("Erreur chargement des données globales.")
        st.stop()

    all_data = pd.DataFrame(data_response.json())
    if 'index' in all_data.columns:
        all_data.set_index('index', inplace=True)
    has_target = 'TARGET' in all_data.columns

    # --- Données Client ---
    st.subheader("📋 Données du client sélectionné")
    client_data = all_data.loc[selected_client_id].copy()
    st.dataframe(client_data.T)

    if has_target and 'TARGET' in client_data.index:
        client_data = client_data.drop("TARGET")

    # --- Valeurs SHAP ---
    shap_response = requests.post(f"{API_BASE}/explain_full", json={"client_id": selected_client_id})
    if shap_response.status_code != 200:
        st.error("Erreur lors de la récupération des valeurs SHAP.")
        st.stop()

    shap_values = pd.Series(shap_response.json())
    top5 = shap_values.abs().sort_values(ascending=False).head(5)
    top5_features = top5.index.tolist()

    st.subheader("🔍 Variables les plus influentes (Top 5 SHAP)")
    st.dataframe(pd.DataFrame({
        "Variable": top5.index,
        "Valeur SHAP": shap_values[top5.index].values
    }).reset_index(drop=True))

    # --- SHAP bar plot (TOP 5) ---
    st.subheader("📊 SHAP Summary Plot")
    fig_shap, ax = plt.subplots()
    explainer = shap.Explanation(
        values=shap_values.values,
        base_values=0.5,
        data=client_data.values,
        feature_names=client_data.index.tolist()
    )
    shap.plots.bar(explainer, max_display=5, show=False)
    st.pyplot(fig_shap)
    plt.clf()

    # --- Boxplots empilés des Top 5 variables SHAP par classe ---
    st.subheader("📦 Boxplots des 5 variables SHAP les plus influentes par classe (empilés)")

    if not has_target:
        st.warning("La colonne TARGET est requise pour afficher les distributions par classe.")
    else:
        class_names = {0: "Crédit Accordé", 1: "Défaut de paiement"}
        colors = {0: 'blue', 1: 'red'}

        for cls in [0, 1]:
            st.markdown(f"### Classe {cls} : {class_names[cls]}")
            filtered_data = all_data[all_data['TARGET'] == cls]

            fig = go.Figure()

            for feat in top5_features:
                fig.add_trace(go.Box(
                    y=filtered_data[feat],
                    name=feat,
                    boxpoints='outliers',
                    marker_color=colors[cls],
                    boxmean=True,
                    line=dict(width=1),
                    opacity=0.6
                ))

                # Ajouter la valeur client pour cette variable
                fig.add_trace(go.Scatter(
                    x=[feat],
                    y=[client_data[feat]],
                    mode='markers',
                    marker=dict(size=12, color='black', symbol='circle'),
                    name=f"Valeur client - {feat}",
                    showlegend=False
                ))

            fig.update_layout(
                yaxis_title="Valeur",
                xaxis_title="Variable",
                boxmode="group",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    # --- Moyennes par classe + client ---
    if has_target:
        st.subheader("📈 Moyenne des variables par classe & client")
        class_0_avg = all_data[all_data["TARGET"] == 0][top5_features].mean()
        class_1_avg = all_data[all_data["TARGET"] == 1][top5_features].mean()
        client_vals = client_data[top5_features]

        mean_comparison = pd.DataFrame({
            "Crédit Accordé (Moy)": class_0_avg,
            "Défaut de paiement (Moy)": class_1_avg,
            f"Client {selected_client_id}": client_vals
        })
        st.dataframe(mean_comparison.style.highlight_max(axis=1, color="lightgreen"))
