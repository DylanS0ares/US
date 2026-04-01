import pandas as pd
import numpy as np
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
import sagemaker
from sagemaker import get_execution_role
from botocore.exceptions import NoCredentialsError, ClientError
import logging
from datetime import datetime
from scipy.spatial import cKDTree

# =====================================================================
# 1. CONFIGURAÇÕES INICIAIS E AWS
# =====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar a sessão do SageMaker e AWS
try:
    role = sagemaker.get_execution_role()
    region = boto3.session.Session().region_name
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    session = sagemaker.Session()
    bucket_name = 'projeto-us-validacao-william'

    print(f"<role>: {role}")
    print(f"<region>: {region}")
    print(f"<account_id>: {account_id}")
    print(f"<bucket>: {bucket_name}")
except Exception as e:
    print(f"Aviso: Não foi possível carregar as credenciais da AWS. {e}")

# Variável global para armazenar as densidades
densidade_df = pd.DataFrame(columns=['imagem', 'densidade'])

# =====================================================================
# 2. FUNÇÕES DE INGESTÃO E PRÉ-PROCESSAMENTO
# =====================================================================
def concatenar_arquivos(path):
    """Concatena todos os arquivos CSV num diretório em um único DataFrame."""
    cont = 0
    arquivos_csv = [arquivo for arquivo in os.listdir(path) if arquivo.endswith('.CSV')]

    if not arquivos_csv:
        print(f"Nenhum arquivo CSV encontrado no diretório: {path}")
        return None

    try:
        lista_dataframes = []
        for arquivo in arquivos_csv:
            caminho_completo = os.path.join(path, arquivo)
            df = pd.read_csv(caminho_completo)
            cont += len(df)
            lista_dataframes.append(df)

        dataframe_concatenado = pd.concat(lista_dataframes, ignore_index=True)
        return dataframe_concatenado

    except Exception as e:
        print(f"Ocorreu um erro durante a concatenação: {e}")
        return None

def preprocessar_e_separar_lados(df):
    """Ajusta a odometria e separa os dados em Trilho Esquerdo e Direito."""
    df['odo'] = df['odo'] * 1000000
    df['odo'] = df['odo'].astype(int)
    
    # Separação com base nas sondas
    df_esq = df[df['probe'].isin([0, 6, 8, 4, 10])]
    df_dir = df[df['probe'].isin([1, 7, 9, 5, 11])]
    
    # Filtro de Amplitude (Grass Noise)
    df_esq = df_esq[df_esq['level'] > 450]
    df_dir = df_dir[df_dir['level'] > 450]
    
    return df_esq, df_dir

# =====================================================================
# 3. FUNÇÕES DE PROCESSAMENTO ESTATÍSTICO E FILTRAGEM
# =====================================================================
def calcular_estatisticas(df):
    """Calcula a estatística T-test por Frame e Probe para validação do sinal."""
    grouped = df.groupby(['frame', 'probe'])
    results = []

    for (frame, probe), group in grouped:
        levels = group['level'].values
        n = len(levels)
        mean = np.mean(levels)
        stddev = np.std(levels, ddof=1) if n > 1 else 0

        if n >= 2:
            t_stat = (mean - 0) / (stddev / np.sqrt(n)) if stddev != 0 else 0
        else:
            t_stat = None 

        results.append({
            'frame': frame, 'probe': probe, 'N': n,
            'Mean': mean, 'StdDev': stddev, 'Ttest': t_stat
        })

    df_est = pd.DataFrame(results)
    df_est.sort_values(by='frame', ascending=False, inplace=True)
    df_est.reset_index(drop=True, inplace=True)

    # Preencher Ttest para N == 1 calculando a derivada discreta
    for i in range(len(df_est)):
        if df_est.loc[i, 'N'] == 1:
            if i + 1 < len(df_est):
                next_mean = df_est.loc[i + 1, 'Mean']
                current_mean = df_est.loc[i, 'Mean']
                df_est.loc[i, 'Ttest'] = next_mean - current_mean
            else:
                df_est.loc[i, 'Ttest'] = 0

    return df_est

def filtrar_df_por_ttest(df, df_est, ttest_limite):
    """Retém apenas as anomalias onde o T-test <= limite."""
    print(f"   -> Filtrando dados: mantendo apenas sinais com T-test <= {ttest_limite}")
    df_est_filtrado = df_est[df_est['Ttest'] <= ttest_limite]
    pares_filtrados = df_est_filtrado[['frame', 'probe']].drop_duplicates()
    df_filtrado = df.merge(pares_filtrados, on=['frame', 'probe'])
    return df_filtrado

def completa_linhas(df):
    """
    Versão Otimizada e Vetorizada.
    Remove ecos estruturais constantes (linhas horizontais) identificando sequências
    de pontos na mesma profundidade ('depth') cuja distância de odometria seja <= 2.
    """
    # 1. Remoção de duplicatas exatas na mesma coordenada (odo, depth) para sondas específicas
    probes_to_check = [6, 8, 4, 10, 7, 9, 5, 11]
    mask_probes = df['probe'].isin(probes_to_check)
    
    # Aplicar o drop_duplicates apenas nas sondas de interesse
    df_to_check = df[mask_probes].drop_duplicates(subset=['odo', 'depth'], keep='first')
    df_others = df[~mask_probes]
    
    # Juntar os dados limpos e ordenar por profundidade e odometria
    df_clean = pd.concat([df_to_check, df_others], ignore_index=True)
    df_sorted = df_clean.sort_values(by=['depth', 'odo']).reset_index(drop=True)
    
    # 2. Identificação Matemática das Linhas Horizontais
    # A função diff() calcula a distância para o vizinho sem precisar de um loop 'while'
    diff_anterior = df_sorted.groupby('depth')['odo'].diff()
    diff_proximo = df_sorted.groupby('depth')['odo'].diff(-1).abs()
    
    # 3. Marcamos os pontos que estão a uma distância <= 2 do seu vizinho
    # fillna(float('inf')) garante que as bordas de cada grupo não deem falso positivo
    is_seq_anterior = diff_anterior.fillna(float('inf')) <= 2
    is_seq_proximo = diff_proximo.fillna(float('inf')) <= 2
    
    # 4. Um ponto é "ruído estrutural" se fizer parte de uma sequência
    mask_remover = is_seq_anterior | is_seq_proximo
    
    # 5. Filtrar o DataFrame mantendo APENAS o que NÃO é linha horizontal (~ mask)
    df_filtrado = df_sorted[~mask_remover]
    
    # 6. Reordenar por 'odo' para que a sequência temporal/espacial volte ao normal para o plot
    df_final = df_filtrado.sort_values(by='odo').reset_index(drop=True)
    
    return df_final
def plot_interval_recortes_local(df, odo_start, odo_end, lado, pasta_base_local):
    """
    Gera os B-Scans e fatia a imagem (Boleto, Alma e Patim)
    mantendo os parâmetros visuais originais e filtrando imagens quase vazias.
    """
    global densidade_df

    df = df.sort_values(by='odo')
    df_interval = df[(df['odo'] >= odo_start) & (df['odo'] <= odo_end)]

    # Cálculo dos pontos
    num_pontos = len(df_interval)
    intervalo_odo = odo_end - odo_start + 1
    densidade = num_pontos / intervalo_odo if intervalo_odo > 0 else 0
    
    # =================================================================
    # CORREÇÃO DO FILTRO: 
    # Agora barra imagens que tenham 3 PONTOS ou menos no total da janela.
    # (Se você quiser barrar por densidade real, use algo como 0.002)
    # =================================================================
    if num_pontos <= 3:
        return 

    densidade_str = f"{densidade:.3f}".replace('.', '_')
    imagem_nome = f"{odo_start}_{odo_end}_{lado}_dens{densidade_str}.jpg"

    # Atualiza o DataFrame global de forma segura
    novo_registro = pd.DataFrame({'imagem': [imagem_nome], 'densidade': [densidade]})
    if densidade_df.empty:
        densidade_df = novo_registro
    else:
        densidade_df = pd.concat([densidade_df, novo_registro], ignore_index=True)

    # Cores mapeadas para as sondas 
    probes_list = [0, 1, 6, 8, 4, 10, 7, 9, 5, 11]
    all_distinct_colors = ['yellow','yellow','green','purple','red','blue','green','purple','red','blue']
    probe_colors = {probe: all_distinct_colors[i] for i, probe in enumerate(probes_list)}

    # Limites anatômicos em profundidade
    limites_profundidade = {
        'boleto': (0, 52),
        'alma': (53, 179),
        'patim': (180, 223)
    }

    # Iterar para salvar o recorte de cada seção
    for secao, (depth_min, depth_max) in limites_profundidade.items():
        
        plt.figure(figsize=(50, 8))
        plt.xlim(odo_start, odo_end)

        if not df_interval.empty:
            sns.scatterplot(data=df_interval, x="odo", y="depth",
                            hue="probe", 
                            s=70, 
                            alpha=1,
                            palette=probe_colors, 
                            legend=False,
                            marker='^', 
                            edgecolor='none', 
                            antialiased=False)

        # Fatiamento do Eixo Y
        plt.ylim(depth_max, depth_min) 
        plt.axis('off')

        # Criação dos caminhos locais
        local_secao_dir = os.path.join(pasta_base_local, secao)
        local_file_path = os.path.join(local_secao_dir, imagem_nome)
        
        # Salvamento
        plt.savefig(local_file_path, bbox_inches='tight')
        plt.close()


def grava_imagens_compactadas(df, lado, tipo):
    """Itera sobre janelas de 2400mm, gera as subpastas, compacta num ZIP e envia para o S3."""
    window_size = 2400
    odo_start = df['odo'].iloc[0]
    odo_end = odo_start + window_size - 1

    # Estrutura de pastas temporárias
    base_tmp_dir = f"/tmp/processamento_{tipo}_{lado}"
    for secao in ['boleto', 'alma', 'patim']:
        os.makedirs(os.path.join(base_tmp_dir, secao), exist_ok=True)

    print(f"A gerar imagens em blocos de {window_size}mm para as subpastas em {base_tmp_dir}...")

    # Variável para rastrear se alguma imagem foi de fato gerada
    arquivos_gerados = 0 

    while odo_start <= df['odo'].iloc[-1]:
        df_interval = df[(df['odo'] >= odo_start) & (df['odo'] <= odo_end)]

        # CORREÇÃO DO FILTRO DE SONDA: Agora processa se houver QUALQUER dado válido na janela
        if not df_interval.empty:
            plot_interval_recortes_local(df_interval, odo_start, odo_end, lado, base_tmp_dir)
            arquivos_gerados += 1

        odo_start = odo_end + 1
        odo_end = odo_start + window_size - 1

    if arquivos_gerados == 0:
        print(f"Aviso: Nenhuma imagem válida encontrada para o lado {lado} após os filtros.")

    # Compactação
    zip_base_name = f"/tmp/{tipo}_{lado}_imagens"
    print(f"A compactar pastas em {zip_base_name}.zip...")
    shutil.make_archive(zip_base_name, 'zip', base_tmp_dir)
    zip_file_path = f"{zip_base_name}.zip"

    # Upload
    s3_key = f"Imagens_Processadas/{tipo}/{tipo}_{lado}_imagens.zip"
    upload_to_s3(zip_file_path, bucket_name, s3_key)

    # Limpeza
    shutil.rmtree(base_tmp_dir)
    os.remove(zip_file_path)

    # Salvar relatório global de densidade
    relatorio_path = 'Concatenados/relatorio_densidade.csv'
    os.makedirs(os.path.dirname(relatorio_path), exist_ok=True)
    densidade_df.to_csv(relatorio_path, index=False)

def remover_pontos_isolados(df, raio=10):
    """
    Remove pontos que não possuem nenhum outro ponto vizinho 
    dentro de um raio Euclidiano especificado (em mm).
    """
    if df.empty:
        return df
        
    print(f"   -> Removendo pontos isolados (Raio de busca: {raio} mm)...")
    
    # Extrai as coordenadas X (odo) e Y (depth) para o espaço 2D
    coords = df[['odo', 'depth']].values
    
    # Cria a árvore de busca espacial (KD-Tree)
    tree = cKDTree(coords)
    
    # Busca os vizinhos para cada ponto dentro do raio estipulado
    # A função retorna uma lista de índices para cada ponto
    vizinhos = tree.query_ball_point(coords, r=raio)
    
    # Conta a quantidade de vizinhos. 
    # Todo ponto é vizinho de si mesmo, então isolados terão len == 1
    num_vizinhos = np.array([len(v) for v in vizinhos])
    
    # Mantém apenas os pontos que têm mais de 1 vizinho (ele mesmo + pelo menos 1 outro)
    df_filtrado = df[num_vizinhos > 1].copy()
    
    pontos_removidos = len(df) - len(df_filtrado)
    print(f"      * {pontos_removidos} pontos isolados foram filtrados e removidos.")
    
    return df_filtrado.reset_index(drop=True)    
# =====================================================================
# 4. FUNÇÕES DE RENDERIZAÇÃO, FATIAMENTO E UPLOAD S3
# =====================================================================
def upload_to_s3(file_path, bucket_name, s3_key):
    """Envia um arquivo para o bucket do S3."""
    try:
        s3_client = boto3.client('s3', region_name=region)
        s3_client.upload_file(file_path, bucket_name, s3_key)
        logging.info(f"Arquivo '{file_path}' enviado para s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logging.error(f"Erro no upload para S3: {e}")



def grava_imagens_compactadas(df, lado, tipo):
    """Itera sobre janelas de 2400mm, gera as subpastas, compacta num ZIP e envia para o S3 com progresso."""
    window_size = 2400
    odo_start = df['odo'].iloc[0]
    odo_max = df['odo'].iloc[-1]
    odo_end = odo_start + window_size - 1

    # Cálculo estimado do total de iterações (janelas)
    total_janelas = int((odo_max - odo_start) / window_size) + 1

    # Estrutura de pastas temporárias
    base_tmp_dir = f"/tmp/processamento_{tipo}_{lado}"
    for secao in ['boleto', 'alma', 'patim']:
        os.makedirs(os.path.join(base_tmp_dir, secao), exist_ok=True)

    print(f"A gerar imagens para {tipo} (Lado: {lado}) em {base_tmp_dir}...")
    print(f"Total de janelas de {window_size}mm a analisar: {total_janelas}")

    arquivos_gerados = 0 
    janela_atual = 0
    passo_print = max(1, int(total_janelas / 10)) # Calcula o intervalo para printar a cada ~10%

    while odo_start <= odo_max:
        janela_atual += 1
        
        # =================================================================
        # NOVO: Print de Progresso a cada 10% ou na última janela
        # =================================================================
        if janela_atual % passo_print == 0 or janela_atual == total_janelas:
            percentual = (janela_atual / total_janelas) * 100
            # Como cada janela válida gera 3 imagens (Boleto, Alma, Patim), multiplicamos por 3
            total_imagens_salvas = arquivos_gerados * 3 
            print(f"   -> Progresso: {percentual:.1f}% ({janela_atual}/{total_janelas} janelas analisadas) | Imagens geradas até agora: {total_imagens_salvas}")

        df_interval = df[(df['odo'] >= odo_start) & (df['odo'] <= odo_end)]

        if not df_interval.empty:
            plot_interval_recortes_local(df_interval, odo_start, odo_end, lado, base_tmp_dir)
            arquivos_gerados += 1

        odo_start = odo_end + 1
        odo_end = odo_start + window_size - 1

    if arquivos_gerados == 0:
        print(f"Aviso: Nenhuma imagem válida encontrada para o lado {lado} após os filtros.")
    else:
        print(f"Geração concluída! Total de janelas com dados: {arquivos_gerados} ({arquivos_gerados * 3} imagens criadas).")

    # Compactação
    zip_base_name = f"/tmp/{tipo}_{lado}_imagens"
    print(f"A compactar pastas em {zip_base_name}.zip...")
    shutil.make_archive(zip_base_name, 'zip', base_tmp_dir)
    zip_file_path = f"{zip_base_name}.zip"

    # Upload
    s3_key = f"Imagens_Processadas/{tipo}/{tipo}_{lado}_imagens.zip"
    print(f"Enviando para o S3: s3://{bucket_name}/{s3_key}")
    upload_to_s3(zip_file_path, bucket_name, s3_key)

    # Limpeza
    shutil.rmtree(base_tmp_dir)
    os.remove(zip_file_path)

    # Salvar relatório global de densidade
    relatorio_path = 'Concatenados/relatorio_densidade.csv'
    os.makedirs(os.path.dirname(relatorio_path), exist_ok=True)
    densidade_df.to_csv(relatorio_path, index=False)

# =====================================================================
# 5. BLOCO DE EXECUÇÃO PRINCIPAL (MAIN PIPELINE)
# =====================================================================
if __name__ == "__main__":
    
    # 1. DEFINA AQUI O DIRETÓRIO DOS SEUS CSVs BRUTOS
    diretorio_csvs_brutos = 'dados/GRM_FPT_12_03_2024-BHC'
    
    # -----------------------------------------------------------------
    # NOVO: Extrair as 3 últimas letras do nome da pasta
    # os.path.normpath remove barras finais acidentais (ex: 'pasta/')
    # os.path.basename pega apenas o nome final da pasta
    # [-3:] fatia a string pegando os 3 últimos caracteres
    # -----------------------------------------------------------------
    nome_pasta = os.path.basename(os.path.normpath(diretorio_csvs_brutos))
    tipo_dinamico = nome_pasta[-3:] # Neste caso, resultará em 'BHC'
    
    print(f"Iniciando Pipeline de Processamento (Tipo Extraído: {tipo_dinamico})...")
    
    # Etapa 1: Concatenação
    print("1. Concatenando Arquivos...")
    df_completo = concatenar_arquivos(diretorio_csvs_brutos)
    
    if df_completo is not None:
        # Etapa 2: Pré-processamento e Separação por Lado
        print("2. Pré-processando e separando trilhos...")
        df_completo = df_completo.sort_values(by='odo')
        df_esq, df_dir = preprocessar_e_separar_lados(df_completo)
        
        # Etapa 3: Cálculos Estatísticos
        print("3. Calculando Estatísticas e T-test...")
        df_esq_est = calcular_estatisticas(df_esq)
        df_dir_est = calcular_estatisticas(df_dir)
        
        # Etapa 4: Filtragem pelo T-test <= 10
        print("4. Aplicando Limiar do T-test...")
        parametro_ttest = 10
        print(f"4. Aplicando Limiar do T-test (Parâmetro configurado: {parametro_ttest})...")
        df_esq_ttest = filtrar_df_por_ttest(df_esq, df_esq_est, parametro_ttest)
        df_dir_ttest = filtrar_df_por_ttest(df_dir, df_dir_est, parametro_ttest)
       
        
        # Etapa 5: Limpeza de Ecos Estruturais (Linhas Horizontais)
        print("5. Removendo ecos estruturais constantes...")
        df_esq_final = completa_linhas(df_esq_ttest)
        df_dir_final = completa_linhas(df_dir_ttest)

      
        # =========================================================
        # Etapa 5.5: NOVO FILTRO - Remoção de Pontos Isolados
        # =========================================================
        print("5.5 Aplicando filtro espacial de pontos isolados...")
        raio_isolamento = 10 # Define o raio de 10 unidades (mm)
        
        if not df_esq_final.empty:
            print("--- Trilho Esquerdo ---")
            df_esq_final = remover_pontos_isolados(df_esq_final, raio=raio_isolamento)
            
        if not df_dir_final.empty:
            print("--- Trilho Direito ---")
            df_dir_final = remover_pontos_isolados(df_dir_final, raio=raio_isolamento)
        
        
        # Etapa 6: Geração, Fatiamento, Compactação e Upload
        print(f"6. Gerando imagens, fatiando (Boleto/Alma/Patim) e enviando para o S3 ({tipo_dinamico})...")
        
        # Processar Lado Esquerdo - Passando o tipo dinâmico
        print("--- Processando Lado Esquerdo ---")
        if not df_esq_final.empty:
            grava_imagens_compactadas(df_esq_final, lado='esq', tipo=tipo_dinamico)
            
        # Processar Lado Direito - Passando o tipo dinâmico
        print("--- Processando Lado Direito ---")
        if not df_dir_final.empty:
            grava_imagens_compactadas(df_dir_final, lado='dir', tipo=tipo_dinamico)
            
        print("✅ Pipeline executado com sucesso!")
    else:
        print("❌ Pipeline abortado: Falha na leitura dos dados brutos.")