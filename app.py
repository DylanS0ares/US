import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
import zipfile
import boto3
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

# ================= CONFIG =================
BUCKET_NAME = 'projeto-us-validacao-william'
PREFIXO_S3 = 'Imagens_Processadas/BHC/'

PASTA_IMAGENS = './Imagens_US'
PASTA_RELATORIOS = './relatorios'

# Modelo único, mesmo diretório do app.py
MODELO_YOLO = 'best_alma_1.pt'

LIMITES_DEPTH = {
    'alma': (53, 179)
}

# ================= UI =================
st.title("🚄 Pipeline US + YOLO + AWS")

# =========================================
# FUNÇÕES AWS
# =========================================
def baixar_s3():
    s3 = boto3.client('s3')
    os.makedirs(PASTA_IMAGENS, exist_ok=True)

    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIXO_S3)

    if 'Contents' not in response:
        st.warning("Nenhum arquivo encontrado no S3")
        return

    for obj in response['Contents']:
        if obj['Key'].endswith('.zip'):
            file_path = os.path.join(PASTA_IMAGENS, os.path.basename(obj['Key']))
            s3.download_file(BUCKET_NAME, obj['Key'], file_path)

def extrair_zip():
    for file in os.listdir(PASTA_IMAGENS):
        if file.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(PASTA_IMAGENS, file), 'r') as zip_ref:
                zip_ref.extractall(PASTA_IMAGENS)

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
    # Confirma se o arquivo do modelo existe
    if not os.path.exists(MODELO_YOLO):
        st.error(f"Modelo não encontrado: {MODELO_YOLO}")
        return pd.DataFrame(columns=["imagem", "classe", "secao"])

    model = YOLO(MODELO_YOLO)
    resultados = []

    for root, _, files in os.walk(PASTA_IMAGENS):
        for img_name in files:
            if not img_name.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(root, img_name)
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
        return pd.DataFrame(columns=["imagem", "classe", "secao"])

# =========================================
# BOTÕES
# =========================================
if st.button("📥 Baixar dados do S3"):
    baixar_s3()
    extrair_zip()
    st.success("Download e extração concluídos!")

uploaded_file = st.file_uploader("📂 Envie CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = preprocessar(df)
    df = remover_isolados(df)

    st.write("Dados processados:")
    st.dataframe(df.head())

    if st.button("📊 Gerar imagem"):
        gerar_imagem(df, "saida.png")
        st.image("saida.png")

if st.button("🧠 Rodar YOLO"):
    with st.spinner("Executando inferência YOLO..."):
        df_res = rodar_yolo()
    st.write(df_res)

    os.makedirs(PASTA_RELATORIOS, exist_ok=True)
    path = os.path.join(PASTA_RELATORIOS, f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_res.to_csv(path, index=False)

    st.success(f"Inferência concluída! CSV salvo em {path}")