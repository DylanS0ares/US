import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
import zipfile

# ================= CONFIG =================
PASTA_MODELOS = './'  # modelo está na mesma pasta do app.py
PASTA_IMAGENS = './Imagens_US'
PASTA_RELATORIOS = './relatorios'

MODELO_YOLO = 'best_alma_1.pt'  # nome exato do modelo
LIMITES_DEPTH = {
    'alma': (53, 179)  # apenas a seção alma
}

# ================= UI =================
st.title("🚄 Pipeline US + YOLO Local")

# =========================================
# PROCESSAMENTO CSV
# =========================================
def preprocessar(df):
    df['odo'] = (df['odo'] * 1000000).astype(int)
    df = df[df['level'] > 450]
    return df

def remover_isolados(df):
    coords = df[['odo', 'depth']].values
    tree = cKDTree(coords)
    vizinhos = tree.query_ball_point(coords, r=10)
    mask = np.array([len(v) > 1 for v in vizinhos])
    return df[mask]

def gerar_imagem(df, nome):
    plt.figure(figsize=(10, 4))
    sns.scatterplot(data=df, x="odo", y="depth", s=10)
    plt.gca().invert_yaxis()
    plt.savefig(nome)
    plt.close()

# =========================================
# YOLO
# =========================================
def rodar_yolo():
    model_path = os.path.join(PASTA_MODELOS, MODELO_YOLO)
    if not os.path.exists(model_path):
        st.error(f"Modelo não encontrado: {model_path}")
        return pd.DataFrame()

    resultados = []

    for root, _, files in os.walk(PASTA_IMAGENS):
        for img_name in files:
            if not img_name.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(root, img_name)
            model = YOLO(model_path)
            res = model.predict(img_path, verbose=False)

            for r in res:
                for box in r.boxes:
                    resultados.append({
                        "imagem": img_name,
                        "classe": model.names[int(box.cls)],
                        "secao": "alma"
                    })

    if resultados:
        return pd.DataFrame(resultados)
    else:
        return pd.DataFrame(columns=["imagem","classe","secao"])

# =========================================
# UPLOAD CSV
# =========================================
uploaded_file = st.file_uploader("📂 Envie CSV para processar")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = preprocessar(df)
    df = remover_isolados(df)

    st.write("✅ Dados processados:")
    st.dataframe(df.head())

    if st.button("📊 Gerar gráfico"):
        gerar_imagem(df, "saida.png")
        st.image("saida.png")

# =========================================
# BOTÃO YOLO
# =========================================
if st.button("🧠 Rodar YOLO"):
    if not os.path.exists(PASTA_IMAGENS):
        st.warning(f"Pasta de imagens não encontrada: {PASTA_IMAGENS}")
    else:
        with st.spinner("Executando inferência YOLO..."):
            df_res = rodar_yolo()
            st.write(df_res)

            os.makedirs(PASTA_RELATORIOS, exist_ok=True)
            nome_csv = f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            path = os.path.join(PASTA_RELATORIOS, nome_csv)
            df_res.to_csv(path, index=False)

            st.success(f"Inferência concluída! CSV salvo em {path}")

            # botão para baixar o CSV
            with open(path, "rb") as f:
                st.download_button(
                    label="⬇️ Baixar CSV",
                    data=f,
                    file_name=nome_csv,
                    mime="text/csv"
                )