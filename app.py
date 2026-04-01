import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
import zipfile
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

# ================= CONFIG =================
PASTA_MODELOS = './modelo'
PASTA_IMAGENS = './Imagens_US'
PASTA_RELATORIOS = './relatorios'

# ================= UI =================
st.title("🚄 Pipeline US + YOLO (modelo best_alma_1.pt)")

# =========================================
# FUNÇÕES
# =========================================
def extrair_zip(file):
    os.makedirs(PASTA_IMAGENS, exist_ok=True)
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(PASTA_IMAGENS)
    st.success("ZIP extraído!")

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

def rodar_yolo():
    model_path = os.path.join(PASTA_MODELOS, "best_alma_1.pt")
    model = YOLO(model_path)
    resultados = []

    for root, _, files in os.walk(PASTA_IMAGENS):
        for img_name in files:
            if not img_name.lower().endswith((".jpg",".png")):
                continue

            img_path = os.path.join(root, img_name)
            res = model.predict(img_path, verbose=False)

            for r in res:
                for box in r.boxes:
                    resultados.append({
                        "imagem": img_name,
                        "classe": model.names[int(box.cls)]
                    })

    if resultados:
        return pd.DataFrame(resultados)
    else:
        return pd.DataFrame(columns=["imagem","classe"])

# =========================================
# UPLOAD E PROCESSAMENTO
# =========================================
uploaded_zip = st.file_uploader("📂 Envie ZIP com imagens", type="zip")
if uploaded_zip:
    extrair_zip(uploaded_zip)

uploaded_csv = st.file_uploader("📂 Envie CSV")
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    df = preprocessar(df)
    df = remover_isolados(df)
    st.write("Dados processados:")
    st.dataframe(df.head())

    if st.button("📊 Gerar imagem"):
        gerar_imagem(df, "saida.png")
        st.image("saida.png")

# RODAR YOLO
if st.button("🧠 Rodar YOLO"):
    with st.spinner("Executando inferência YOLO..."):
        df_res = rodar_yolo()
    st.write(df_res)

    os.makedirs(PASTA_RELATORIOS, exist_ok=True)
    path = os.path.join(PASTA_RELATORIOS, f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_res.to_csv(path, index=False)

    # Botão de download
    with open(path, "rb") as f:
        st.download_button(
            label="⬇️ Baixar CSV",
            data=f,
            file_name=os.path.basename(path),
            mime="text/csv"
        )