"""
Módulo de Pré-processamento para Dados de Séries Temporais

Este módulo contém funções para preparar dados de séries temporais para modelos LSTM,
incluindo normalização, criação de sequências e divisão dos dados.

Autor: Fabio Matos
Data: 27/07/2025
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import os
import joblib

# Configurações
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dados')
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'modelos')
os.makedirs(models_dir, exist_ok=True)

def normalizar_dados(dados: pd.DataFrame, 
                   colunas: list = None, 
                   tipo_escalonador: str = 'minmax',
                   salvar_escalonador: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Normaliza os dados usando MinMaxScaler ou StandardScaler.
    
    Parâmetros:
    -----------
    dados : pd.DataFrame
        DataFrame com os dados a serem normalizados
    colunas : list, opcional
        Lista de colunas para normalizar. Se None, normaliza todas as colunas numéricas.
    tipo_escalonador : str, default='minmax'
        Tipo de escalonador a ser usado: 'minmax' ou 'standard'
    salvar_escalonador : bool, default=True
        Se True, salva o escalonador para uso posterior na inversão
        
    Retorna:
    --------
    Tuple[pd.DataFrame, dict]
        - DataFrame com os dados normalizados
        - Dicionário com informações dos escalonadores
    """
    if colunas is None:
        colunas = dados.select_dtypes(include=[np.number]).columns.tolist()
    
    dados_normalizados = dados.copy()
    escalonadores = {}
    
    for col in colunas:
        if tipo_escalonador.lower() == 'minmax':
            escalonador = MinMaxScaler(feature_range=(0, 1))
        else:
            escalonador = StandardScaler()
            
        # Ajusta e transforma os dados
        dados_normalizados[col] = escalonador.fit_transform(dados[[col]])
        
        # Armazena o escalonador para uso posterior
        escalonadores[col] = escalonador
        
        # Salva o escalonador se necessário
        if salvar_escalonador:
            os.makedirs(os.path.join(models_dir, 'escalonadores'), exist_ok=True)
            joblib.dump(
                escalonador, 
                os.path.join(models_dir, 'escalonadores', f'escalonador_{col}.joblib')
            )
    
    return dados_normalizados, escalonadores

def criar_sequencias(dados: np.ndarray, 
                   tamanho_janela: int = 60, 
                   passo: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria sequências de entrada e saída para treinamento de modelos de séries temporais.
    
    Parâmetros:
    -----------
    dados : np.ndarray
        Array com os dados da série temporal
    tamanho_janela : int, default=60
        Número de passos de tempo a serem usados para prever o próximo valor
    passo : int, default=1
        Passo entre as sequências consecutivas
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        - Array com as sequências de entrada (X)
        - Array com os valores alvo (y)
    """
    X, y = [], []
    for i in range(0, len(dados) - tamanho_janela, passo):
        X.append(dados[i:(i + tamanho_janela)])
        y.append(dados[i + tamanho_janela])
    return np.array(X), np.array(y)

def preparar_dados_para_lstm(dados: pd.DataFrame, 
                          coluna_alvo: str = 'Close',
                          tamanho_janela: int = 60,
                          passo: int = 60,
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Prepara os dados para treinamento de um modelo LSTM.
    
    Parâmetros:
    -----------
    dados : pd.DataFrame
        DataFrame com os dados da série temporal
    coluna_alvo : str, default='Close'
        Nome da coluna alvo para previsão
    tamanho_janela : int, default=60
        Tamanho da janela para as sequências
    passo : int, default=1
        Passo entre as sequências consecutivas
    test_size : float, default=0.2
        Proporção dos dados a serem usados como teste
    val_size : float, default=0.1
        Proporção dos dados de treino a serem usados como validação
    random_state : int, default=42
        Semente para reprodutibilidade
        
    Retorna:
    --------
    Dict[str, np.ndarray]
        Dicionário contendo os conjuntos de treino, validação e teste
    """
    # 1. Normalizar os dados
    dados_normalizados, _ = normalizar_dados(dados[[coluna_alvo]])
    
    # 2. Criar sequências
    X, y = criar_sequencias(dados_normalizados[coluna_alvo].values, 
                           tamanho_janela=tamanho_janela, 
                           passo=passo)
    
    # 3. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # 4. Separar validação do treino, se necessário
    if val_size > 0:
        val_size_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size_relative, 
            random_state=random_state,
            shuffle=False
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    else:
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }

def carregar_dados_preparados(caminho: str) -> Dict[str, np.ndarray]:
    """
    Carrega dados já preparados a partir de um arquivo.
    
    Parâmetros:
    -----------
    caminho : str
        Caminho para o arquivo .npz contendo os dados
        
    Retorna:
    --------
    Dict[str, np.ndarray]
        Dicionário com os conjuntos de dados
    """
    return dict(np.load(caminho, allow_pickle=True))

def salvar_dados_preparados(dados: Dict[str, np.ndarray], nome_arquivo: str):
    """
    Salva os dados preparados em um arquivo .npz.
    
    Parâmetros:
    -----------
    dados : Dict[str, np.ndarray]
        Dicionário com os conjuntos de dados
    nome_arquivo : str
        Nome do arquivo para salvar (sem extensão)
    """
    os.makedirs(os.path.join(data_dir, 'preparados'), exist_ok=True)
    caminho = os.path.join(data_dir, 'preparados', f'{nome_arquivo}.npz')
    np.savez_compressed(caminho, **dados)
    print(f"Dados salvos em: {caminho}")

def pipeline_preprocessamento(caminho_dados: str, 
                           coluna_alvo: str = 'Close',
                           tamanho_janela: int = 60,
                           passo: int = 60,
                           test_size: float = 0.2,
                           val_size: float = 0.1,
                           salvar_dados: bool = True) -> Dict[str, np.ndarray]:
    """
    Pipeline completo de pré-processamento para dados de séries temporais.
    
    Parâmetros:
    -----------
    caminho_dados : str
        Caminho para o arquivo CSV com os dados brutos
    coluna_alvo : str, default='Close'
        Nome da coluna alvo para previsão
    tamanho_janela : int, default=60
        Tamanho da janela para as sequências
    test_size : float, default=0.2
        Proporção dos dados a serem usados como teste
    val_size : float, default=0.1
        Proporção dos dados de treino a serem usados como validação
    salvar_dados : bool, default=True
        Se True, salva os dados processados em disco
        
    Retorna:
    --------
    Dict[str, np.ndarray]
        Dicionário contendo os conjuntos de treino, validação e teste
    """
    # 1. Carregar dados
    # Lê o arquivo CSV, pulando as 3 primeiras linhas de cabeçalho
    # e definindo manualmente os nomes das colunas
    colunas = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Retorno']
    dados = pd.read_csv(
        caminho_dados,
        skiprows=3,  # Pula as 3 primeiras linhas de cabeçalho
        header=None,  # Não usa linha de cabeçalho do arquivo
        names=colunas,  # Define os nomes das colunas manualmente
        usecols=range(len(colunas)),  # Garante que todas as colunas sejam lidas
        na_values=['', 'NaN', 'nan', 'None', 'null'],  # Trata valores ausentes
        parse_dates=['Date'],  # Converte a coluna de data automaticamente
        date_format='%Y-%m-%d'  # Formato da data
    )
    
    # Define a coluna de data como índice
    dados.set_index('Date', inplace=True)
    
    # Remove linhas com valores ausentes
    dados = dados.dropna()
    
    # Converte os tipos de dados para economizar memória
    for col in ['Close', 'High', 'Low', 'Open']:
        dados[col] = dados[col].astype('float32')
    dados['Volume'] = dados['Volume'].astype('int64')
    
    # 2. Preparar dados para LSTM
    dados_preparados = preparar_dados_para_lstm(
        dados=dados,
        coluna_alvo=coluna_alvo,
        tamanho_janela=tamanho_janela,
        passo=passo, 
        test_size=test_size,
        val_size=val_size
    )
    
    # 3. Salvar dados processados se necessário
    if salvar_dados:
        nome_arquivo = f'dados_preparados_janela{tamanho_janela}_passo{passo}'
        salvar_dados_preparados(dados_preparados, nome_arquivo)
    
    return dados_preparados

def visualizar_dados_processados(caminho_arquivo: str):
    """
    Carrega e exibe informações sobre os dados processados.
    
    Parâmetros:
    -----------
    caminho_arquivo : str
        Caminho para o arquivo .npz com os dados processados
    """
    print("\nVisualizando dados processados:")
    print("-" * 50)
    
    # Carrega os dados
    dados = np.load(caminho_arquivo, allow_pickle=True)
    
    # Mostra as chaves disponíveis
    print("Chaves disponíveis no arquivo:", list(dados.keys()))
    
    # Mostra o formato de cada conjunto de dados
    for key in dados.keys():
        print(f"\n{key}:")
        print(f"  Formato: {dados[key].shape}")
        print(f"  Tipo: {dados[key].dtype}")
        print(f"  Primeiros valores: {dados[key][0] if len(dados[key].shape) > 0 else dados[key]}")
    
    print("\nPré-visualização dos dados de treino (X_train):")
    print(dados['X_train'][0])
    print("\nPré-visualização dos alvos de treino (y_train):")
    print(dados['y_train'][0])

if __name__ == "__main__":
    import argparse
    
    # Configura o parser de argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Pré-processamento de dados para modelo LSTM')
    parser.add_argument('arquivo', type=str, help='Nome do arquivo CSV com os dados (deve estar na pasta dados)')
    parser.add_argument('--janela', type=int, default=20, help='Tamanho da janela para as sequências (padrão: 20)')
    parser.add_argument('--passo', type=int, default=5, help='Passo entre as sequências consecutivas (padrão: 5)')
    
    # Parse dos argumentos
    args = parser.parse_args()
    
    # Constrói o caminho completo para o arquivo
    caminho_dados = os.path.join(data_dir, args.arquivo)
    tamanho_janela = args.janela
    passo = args.passo
    
    # Verifica se o arquivo existe
    if not os.path.exists(caminho_dados):
        print(f"Erro: Arquivo não encontrado: {caminho_dados}")
        print("Certifique-se de que o arquivo está na pasta dados/")
        exit(1)
    
    print(f"Iniciando pipeline de pré-processamento...")
    print(f"Arquivo: {args.arquivo}")
    print(f"Tamanho da janela: {tamanho_janela}")
    print(f"Passo: {passo}")
    
    try:
        dados_preparados = pipeline_preprocessamento(
            caminho_dados=caminho_dados,
            coluna_alvo='Close',
            tamanho_janela=tamanho_janela,
            passo=passo,
            test_size=0.2,
            val_size=0.1,
            salvar_dados=True
        )
        print("Pré-processamento concluído com sucesso!")
        
        # Visualiza os dados processados
        caminho_arquivo = os.path.join(data_dir, 'preparados', f'dados_preparados_janela{tamanho_janela}_passo{passo}.npz')
        if os.path.exists(caminho_arquivo):
            visualizar_dados_processados(caminho_arquivo)
    except Exception as e:
        print(f"Ocorreu um erro durante o pré-processamento: {str(e)}")
        exit(1)
