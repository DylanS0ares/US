import os
import cv2
import pandas as pd
import numpy as np
import zipfile
import boto3
from datetime import datetime
from ultralytics import YOLO

# =====================================================================
# 1. CONFIGURAÇÕES DE DIRETÓRIOS E AWS
# =====================================================================
BUCKET_NAME = 'projeto-us-validacao-william'
PREFIXO_S3 = 'Imagens_Processadas/BHC/' 

PASTA_MODELOS = './modelo'           
PASTA_IMAGENS = './Imagens_US'       
PASTA_RELATORIOS = './relatorios'    

LIMITES_DEPTH = {
    'boleto': (0, 52),
    'alma': (53, 179),
    'patim': (180, 223)
}

# =====================================================================
# 2. FUNÇÕES AUXILIARES E INTEGRAÇÃO AWS
# =====================================================================
def baixar_zips_do_s3(bucket, prefixo, pasta_destino):
    s3_client = boto3.client('s3')
    os.makedirs(pasta_destino, exist_ok=True)
    
    print(f"\n☁️ Conectando ao S3... Buscando em s3://{bucket}/{prefixo}")
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefixo)
        
        if 'Contents' not in response:
            print("⚠️ Nenhum arquivo encontrado neste caminho do S3.")
            return

        zips_baixados = 0
        for obj in response['Contents']:
            chave_s3 = obj['Key']
            if chave_s3.lower().endswith('.zip'):
                nome_arquivo = os.path.basename(chave_s3)
                caminho_local = os.path.join(pasta_destino, nome_arquivo)
                
                print(f"⬇️ Baixando: {nome_arquivo} ({obj['Size'] / 1024 / 1024:.2f} MB)...")
                s3_client.download_file(bucket, chave_s3, caminho_local)
                zips_baixados += 1
                
        if zips_baixados > 0:
            print(f"✅ Download do S3 concluído! ({zips_baixados} arquivo(s) baixado(s)).")
            
    except Exception as e:
        print(f"❌ Erro ao acessar o S3. Verifique as permissões. Detalhe: {e}")

def descompactar_arquivos_zip(pasta_alvo):
    if not os.path.exists(pasta_alvo):
        return

    zips_encontrados = [f for f in os.listdir(pasta_alvo) if f.lower().endswith('.zip')]
    
    for arquivo_zip in zips_encontrados:
        caminho_zip = os.path.join(pasta_alvo, arquivo_zip)
        print(f"\n📦 Extraindo arquivo: '{arquivo_zip}'...")
        
        try:
            with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                zip_ref.extractall(pasta_alvo)
            
            os.remove(caminho_zip) 
            print(f"✅ Extração concluída. Arquivo original removido.")
        except zipfile.BadZipFile:
            print(f"❌ Erro: O arquivo '{arquivo_zip}' está corrompido.")

def mapear_modelos_disponiveis(pasta_modelos):
    modelos_encontrados = {}
    secoes_alvo = ['boleto', 'alma', 'patim']
    
    if not os.path.exists(pasta_modelos):
        print(f"⚠️ Aviso: A pasta de modelos '{pasta_modelos}' não foi encontrada.")
        return modelos_encontrados

    for arquivo in os.listdir(pasta_modelos):
        if arquivo.endswith('.pt'):
            nome_min = arquivo.lower()
            for secao in secoes_alvo:
                if secao in nome_min:
                    modelos_encontrados[secao] = os.path.join(pasta_modelos, arquivo)
                    print(f"🧠 Modelo para '{secao.capitalize()}' mapeado: {arquivo}")
                    
    return modelos_encontrados

def grava_relatorio(df, nome_base):
    if df.empty:
        print(f"⚠️ Nenhum defeito encontrado para {nome_base}. Relatório ignorado.")
        return

    os.makedirs(PASTA_RELATORIOS, exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_base_name = f'{nome_base}_{current_time}'

    csv_file_path = os.path.join(PASTA_RELATORIOS, f'{file_base_name}.csv')
    xlsx_file_path = os.path.join(PASTA_RELATORIOS, f'{file_base_name}.xlsx')

    df.to_csv(csv_file_path, index=False)
    df.to_excel(xlsx_file_path, index=False)
    print(f"📄 Relatório gerado: {PASTA_RELATORIOS}/{file_base_name} (.csv/.xlsx)")

# =====================================================================
# 3. PIPELINE DE INFERÊNCIA
# =====================================================================
def executar_inferencia():
    print("=====================================================")
    print(" INICIANDO PIPELINE DE INFERÊNCIA (S3 -> YOLO -> XLS) ")
    print("=====================================================")
    
    baixar_zips_do_s3(BUCKET_NAME, PREFIXO_S3, PASTA_IMAGENS)
    descompactar_arquivos_zip(PASTA_IMAGENS)
    
    print("\nVerificando modelos de IA disponíveis...")
    modelos_ativos = mapear_modelos_disponiveis(PASTA_MODELOS)
    
    if not modelos_ativos:
        print("❌ Nenhum modelo de secção ('alma', 'boleto', 'patim') encontrado. Fim da execução.")
        return

    pastas_para_processar = []
    for root, dirs, files in os.walk(PASTA_IMAGENS):
        for d in dirs:
            if d.lower() in ['boleto', 'alma', 'patim']:
                pastas_para_processar.append((d.lower(), os.path.join(root, d)))

    for secao, caminho_pasta in pastas_para_processar:
        if secao not in modelos_ativos:
            print(f"⏭️ Pulando pasta '{caminho_pasta}' -> Sem modelo associado.")
            continue
            
        print(f"\n🔍 Analisando pasta: {caminho_pasta}")
        modelo = YOLO(modelos_ativos[secao])
        depth_min, depth_max = LIMITES_DEPTH[secao]
        
        df_resultados = pd.DataFrame(columns=[
            'Localização da Imagem', 'Classe', 'Coordenada ODO(mm)', 
            'Coordenada Depth(mm)', 'Comprimento(mm)'
        ])
        
        image_files = [f for f in os.listdir(caminho_pasta) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        total_imagens = len(image_files)
        
        if total_imagens == 0:
            print("   -> Pasta vazia. Nenhuma imagem para processar.")
            continue
            
        print(f"   -> Total de imagens encontradas: {total_imagens}")
        
        # Define o intervalo para imprimir o progresso (ex: a cada 10%)
        passo_print = max(1, int(total_imagens / 10))
        defeitos_encontrados = 0
        
        # Loop de inferência com contador (enumerate)
        for idx, image_file in enumerate(image_files, start=1):
            
            # Print de progresso
            if idx % passo_print == 0 or idx == total_imagens:
                percentual = (idx / total_imagens) * 100
                print(f"   ⏳ Progresso {secao.capitalize()}: {percentual:.1f}% ({idx}/{total_imagens} imagens) | Defeitos detectados: {defeitos_encontrados}")

            image_path = os.path.join(caminho_pasta, image_file)
            img = cv2.imread(image_path)
            
            if img is None:
                continue
                
            height, width, _ = img.shape
            
            # Inferência
            results = modelo.predict(img, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    defeitos_encontrados += 1
                    class_name = modelo.names[int(box.cls)]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    percentage_x = center_x / width
                    percentage_y = center_y / height
                    
                    x1_perc = x1 / width
                    x2_perc = x2 / width
                    y1_perc = y1 / height
                    y2_perc = y2 / height
                    
                    try:
                        partes_nome = image_file.split('_')
                        initial_x1 = int(partes_nome[0])
                        initial_x2 = int(partes_nome[1])
                    except ValueError:
                        continue
                    
                    coordenada_odo = initial_x1 + int((initial_x2 - initial_x1) * percentage_x)
                    coordenada_depth = depth_min + int((depth_max - depth_min) * percentage_y)
                    
                    x1_t = initial_x1 + int((initial_x2 - initial_x1) * x1_perc)
                    x2_t = initial_x1 + int((initial_x2 - initial_x1) * x2_perc)
                    y1_t = depth_min + int((depth_max - depth_min) * y1_perc)
                    y2_t = depth_min + int((depth_max - depth_min) * y2_perc)
                    
                    comprimento = int(np.sqrt((x2_t - x1_t)**2 + (y2_t - y1_t)**2))
                    
                    novo_dado = pd.DataFrame([{
                        'Localização da Imagem': image_file,
                        'Classe': class_name,
                        'Coordenada ODO(mm)': coordenada_odo,
                        'Coordenada Depth(mm)': coordenada_depth,
                        'Comprimento(mm)': comprimento
                    }])
                    df_resultados = pd.concat([df_resultados, novo_dado], ignore_index=True)
                    
        print(f"✅ Análise de '{secao.capitalize()}' concluída! Total de defeitos: {defeitos_encontrados}")
        
        # Gravação condicional por pasta
        nome_relatorio = f"Relatorio_Inferencia_{secao.capitalize()}"
        grava_relatorio(df_resultados, nome_relatorio)

if __name__ == "__main__":
    executar_inferencia()