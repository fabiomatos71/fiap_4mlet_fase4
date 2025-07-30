"""
Coleta de Dados - Tech Challenge Fase 4
=======================================

Este script realiza a coleta de dados históricos da Disney (DIS) usando a biblioteca yfinance.
O objetivo é obter dados para treinamento do modelo LSTM de previsão de preços.

Autor: Fabio Matos
Data: 21/07/2025
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np
import seaborn as sns

# Configurações
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12

def criar_diretorio_se_nao_existir(diretorio):
    """Cria o diretório especificado se ele não existir."""
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
        print(f"Diretório '{diretorio}' criado com sucesso.")

def baixar_dados_historicos(simbolo='DIS', anos_historico=5):
    """
    Baixa os dados históricos da Disney (DIS).
    
    Parâmetros:
    -----------
    simbolo : str
        Símbolo da ação (padrão: 'DIS' para Disney)
    anos_historico : int
        Número de anos de histórico a serem baixados
        
    Retorna:
    --------
    pandas.DataFrame
        DataFrame com os dados históricos
    """
    # Calcula as datas
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=365 * anos_historico)
    data_inicio_str = data_inicio.strftime('%Y-%m-%d')
    data_fim_str = data_fim.strftime('%Y-%m-%d')
    
    print(f"\nBaixando dados históricos de {simbolo} de {data_inicio_str} até {data_fim_str}...")
    
    try:
        # Baixa os dados usando yfinance
        df = yf.download(
            simbolo,
            start=data_inicio_str,
            end=data_fim_str,
            progress=True
        )
        
        # Verifica se os dados foram baixados corretamente
        if df.empty:
            raise ValueError("Nenhum dado foi retornado. Verifique o símbolo da ação e as datas.")
            
        print(f"\nDados baixados com sucesso!")
        print(f"Período: {df.index[0].date()} a {df.index[-1].date()}")
        print(f"Total de registros: {len(df)}")
        
        return df
    except Exception as e:
        print(f"\nErro ao baixar os dados: {e}")
        return None

def analisar_dados(df, simbolo='DIS'):
    """
    Realiza análise detalhada dos dados coletados.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados históricos
    simbolo : str
        Símbolo da ação para referência
    """
    if df is None or df.empty:
        print("Sem dados para análise!")
        return
    
    print("\n" + "="*50)
    print(f"ANÁLISE DOS DADOS - {simbolo}")
    print("="*50)
    
    # Estatísticas básicas
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    # Verificação de dados faltantes
    print("\nVerificação de dados faltantes:")
    print(df.isnull().sum())
    
    # Cálculo de retornos diários
    if isinstance(df.columns, pd.MultiIndex):
        close_prices = df[('Close', simbolo)] if ('Close', simbolo) in df.columns else df['Close']
        volume = df[('Volume', simbolo)] if ('Volume', simbolo) in df.columns else df['Volume']
    else:
        close_prices = df['Close']
        volume = df['Volume']
    
    df['Retorno'] = close_prices.pct_change()
    
    # Plotagem dos preços e volumes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Gráfico de preços
    ax1.plot(df.index, close_prices, label='Preço de Fechamento', color='blue')
    ax1.set_title(f'Preço de Fechamento - {simbolo}', fontsize=14)
    ax1.set_ylabel('Preço (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico de volume
    ax2.bar(df.index, volume.values, color='gray', alpha=0.7, label='Volume')
    ax2.set_title('Volume de Negociação', fontsize=14)
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Volume')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Ajusta o layout e salva
    plt.tight_layout()
    
    # Cria diretório para imagens se não existir
    criar_diretorio_se_nao_existir('imagens')
    
    # Salva os gráficos
    plt.savefig(f'imagens/analise_{simbolo}.png')
    plt.close()
    
    # Retornos diários
    plt.figure(figsize=(14, 5))
    retornos = df['Retorno'] if 'Retorno' in df.columns else df[('Retorno', '')] if ('Retorno', '') in df.columns else None
    if retornos is not None:
        plt.plot(df.index, retornos, label='Retorno Diário', color='green', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.title(f'Retornos Diários - {simbolo}', fontsize=14)
        plt.xlabel('Data')
        plt.ylabel('Retorno')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'imagens/retornos_{simbolo}.png')
        plt.close()
        
        # Histograma de retornos
        plt.figure(figsize=(14, 5))
        sns.histplot(retornos.dropna(), bins=50, kde=True, color='blue')
        plt.title(f'Distribuição dos Retornos Diários - {simbolo}', fontsize=14)
        plt.xlabel('Retorno')
        plt.ylabel('Frequência')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'imagens/distribuicao_retornos_{simbolo}.png')
        plt.close()
    else:
        print("Aviso: Não foi possível gerar os gráficos de retorno - coluna 'Retorno' não encontrada.")

def salvar_dados(df, simbolo='DIS'):
    """
    Salva os dados coletados em um arquivo CSV.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados históricos
    simbolo : str
        Símbolo da ação para nome do arquivo
        
    Retorna:
    --------
    str
        Caminho do arquivo salvo ou None em caso de erro
    """
    if df is None or df.empty:
        print("Aviso: Nenhum dado para salvar!")
        return None
    
    try:
        # Cria diretório de dados se não existir
        criar_diretorio_se_nao_existir('dados')
        
        # Define o caminho do arquivo
        data_atual = datetime.now().strftime('%Y%m%d')
        arquivo = f'dados/historico_{simbolo}_{data_atual}.csv'
        
        # Salva os dados em CSV
        df.to_csv(arquivo, index_label='Date')
        print(f"\nDados salvos com sucesso em: {os.path.abspath(arquivo)}")
        
        # Salva também em formato parquet (mais eficiente para grandes conjuntos de dados)
        arquivo_parquet = f'dados/historico_{simbolo}_{data_atual}.parquet'
        df.to_parquet(arquivo_parquet, index=True)
        print(f"Dados salvos em formato Parquet: {os.path.abspath(arquivo_parquet)}")
        
        return arquivo
        
    except Exception as e:
        print(f"Erro ao salvar os dados: {e}")
        return None

def verificar_qualidade_dados(df, simbolo='DIS'):
    """
    Verifica a qualidade dos dados coletados.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados históricos
    simbolo : str
        Símbolo da ação para referência
        
    Retorna:
    --------
    bool
        True se os dados estiverem com qualidade aceitável, False caso contrário
    """
    if df is None or df.empty:
        print("Erro: Nenhum dado para verificação de qualidade.")
        return False
    
    print("\n" + "="*50)
    print("VERIFICAÇÃO DE QUALIDADE DOS DADOS")
    print("="*50)
    
    # Verifica valores ausentes
    valores_ausentes = df.isnull().sum()
    total_valores = len(df)
    
    print("\nValores ausentes por coluna:")
    print(valores_ausentes)
    
    # Verifica se há valores ausentes
    if valores_ausentes.sum() > 0:
        print("\nAviso: Foram encontrados valores ausentes nos dados.")
        # Preenche valores ausentes com o método de preenchimento para frente
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)  # Para os primeiros valores, se necessário
        print("Valores ausentes foram preenchidos usando o método de preenchimento para frente.")
    else:
        print("\nNenhum valor ausente encontrado nos dados.")
    
    # Verifica se há valores negativos em colunas que não deveriam ter
    colunas_nao_negativas = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Verifica se o DataFrame tem MultiIndex (colunas com níveis)
    if isinstance(df.columns, pd.MultiIndex):
        # Para MultiIndex, precisamos verificar cada coluna individualmente
        for col in colunas_nao_negativas:
            if col in df.columns.get_level_values(0):
                if (df[col] < 0).any().any():
                    print(f"Aviso: Foram encontrados valores negativos na coluna {col}.")
    else:
        # Para DataFrame normal
        for col in colunas_nao_negativas:
            if col in df.columns:
                if (df[col] < 0).any():
                    print(f"Aviso: Foram encontrados valores negativos na coluna {col}.")
    
    # Verifica se há datas faltando (buracos na série temporal)
    if not df.index.is_monotonic_increasing:
        print("Aviso: O índice de data não está em ordem crescente.")
        df.sort_index(inplace=True)
    
    # Verifica se há duplicatas
    if df.index.duplicated().any():
        print("Aviso: Foram encontradas datas duplicadas nos dados.")
        df = df[~df.index.duplicated(keep='first')]
    
    print("\nVerificação de qualidade concluída.")
    return True

def main():
    """
    Função principal que coordena a coleta, análise e salvamento dos dados.
    """
    print("\n" + "="*50)
    print("COLETA E ANÁLISE DE DADOS HISTÓRICOS - DISNEY (DIS)")
    print("="*50)
    
    # Configurações
    simbolo = 'DIS'  # The Walt Disney Company
    anos_historico = 5  # Número de anos de histórico a serem baixados
    
    # Cria diretórios necessários
    for diretorio in ['dados', 'imagens']:
        criar_diretorio_se_nao_existir(diretorio)
    
    # Baixa os dados históricos
    df = baixar_dados_historicos(simbolo, anos_historico)
    
    if df is not None:
        # Verifica a qualidade dos dados
        if verificar_qualidade_dados(df, simbolo):
            # Análise dos dados
            analisar_dados(df, simbolo)
            
            # Salva os dados
            arquivo_salvo = salvar_dados(df, simbolo)
            
            if arquivo_salvo:
                print(f"\nProcesso concluído com sucesso!")
                print(f"Arquivo de dados: {os.path.abspath(arquivo_salvo)}")
                print(f"Gráficos salvos no diretório: {os.path.abspath('imagens')}")
            else:
                print("\nAviso: Ocorreu um erro ao salvar os dados.")
        else:
            print("\nErro: Problemas na qualidade dos dados coletados.")
    else:
        print("\nFalha ao baixar os dados. Verifique sua conexão com a internet e tente novamente.")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
