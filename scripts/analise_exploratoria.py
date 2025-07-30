import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf

# Configuração de estilo para os gráficos
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Criando diretório para salvar as imagens, se não existir
os.makedirs('imagens/analise_exploratoria', exist_ok=True)
os.makedirs('dados/analise_exploratoria', exist_ok=True)

def obter_dados():
    """Função para carregar os dados do arquivo CSV local"""
    print("Carregando dados do arquivo local...")
    try:
        caminho_dados = 'dados/historico_DIS_20250727.csv'
        colunas = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Retorno']
        df = pd.read_csv(
            caminho_dados,
            skiprows=3,  # Pula as 3 primeiras linhas de cabeçalho
            header=None,  # Não usa linha de cabeçalho do arquivo
            names=colunas,  # Define os nomes das colunas manualmente
            usecols=range(len(colunas)),  # Garante que todas as colunas sejam lidas
            na_values=['', 'NaN', 'nan', 'None', 'null'],  # Trata valores ausentes
            parse_dates=['Date'],  # Converte a coluna de data automaticamente
            date_format='%Y-%m-%d'  # Formato da data
        )
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo de dados não encontrado. Verifique o caminho: {caminho_dados}")

def verificar_dados_ausentes(df):
    """Verifica e relata dados ausentes no DataFrame"""
    print("\n1. Verificando dados ausentes:")
    
    # Verificar se o DataFrame tem MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # Para DataFrames com MultiIndex, acessamos os níveis
        total = df.isnull().sum().unstack()
        percentual = (df.isnull().sum() / len(df) * 100).unstack()
        tabela_ausentes = pd.concat({'Total': total, 'Percentual': percentual}, axis=1)
    else:
        # Para DataFrames sem MultiIndex
        total = df.isnull().sum()
        percentual = (df.isnull().sum() / len(df)) * 100
        tabela_ausentes = pd.concat([total, percentual], axis=1, 
                                   keys=['Total', 'Percentual'])
    
    print(tabela_ausentes)
    
    # Salvando a tabela em markdown
    with open('dados/analise_exploratoria/relatorio_ausentes.md', 'w') as f:
        f.write("# Relatório de Dados Ausentes\n\n")
        f.write(tabela_ausentes.to_markdown())
    
    return tabela_ausentes

def gerar_estatisticas_descritivas(df):
    """Gera estatísticas descritivas do DataFrame"""
    print("\n2. Estatísticas Descritivas:")
    
    # Verificar se o DataFrame tem MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # Para DataFrames com MultiIndex, acessamos os níveis
        estatisticas = df.describe().stack(0).T
    else:
        # Para DataFrames sem MultiIndex
        estatisticas = df.describe().T
    
    print(estatisticas)
    
    # Salvando as estatísticas em markdown
    with open('dados/analise_exploratoria/estatisticas_descritivas.md', 'w') as f:
        f.write("# Estatísticas Descritivas\n\n")
        f.write(estatisticas.to_markdown())
    
    return estatisticas

def plotar_serie_temporal(df):
    """Gráfico de série temporal dos preços de fechamento"""
    print("\n3. Gerando gráfico de série temporal...")
    plt.figure(figsize=(14, 7))
    
    try:
        # Verificar se o DataFrame tem MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # Para DataFrames com MultiIndex, acessamos a coluna 'Close' do primeiro nível
            if ('Close', 'DIS') in df.columns:
                precos = df[('Close', 'DIS')].copy()
            else:
                # Se não encontrar exatamente ('Close', 'DIS'), tenta encontrar 'Close' em qualquer nível
                col_close = [col for col in df.columns if 'Close' in str(col)]
                if not col_close:
                    print("Coluna 'Close' não encontrada no DataFrame.")
                    return None
                precos = df[col_close[0]].copy()
        else:
            # Para DataFrames sem MultiIndex
            if 'Close' not in df.columns:
                print("Coluna 'Close' não encontrada no DataFrame.")
                return None
            precos = df['Close'].copy()
        
        # Garantir que estamos trabalhando com valores numéricos
        precos = pd.to_numeric(precos, errors='coerce')
        precos = precos.dropna()
        
        if not precos.empty:
            plt.plot(precos.index, precos.values, label='Preço de Fechamento', color='blue')
            plt.title('Série Temporal do Preço de Fechamento (DIS)', fontsize=16)
            plt.xlabel('Data', fontsize=12)
            plt.ylabel('Preço (USD)', fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Salvando o gráfico
            caminho_grafico = 'imagens/analise_exploratoria/serie_temporal.png'
            plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Gráfico salvo em: {caminho_grafico}")
            return caminho_grafico
    except Exception as e:
        print(f"Erro ao gerar o gráfico de série temporal: {e}")
    
    print("Não foi possível gerar o gráfico de série temporal. Dados insuficientes ou formato incorreto.")
    return None

def plotar_distribuicao(df, coluna='Close'):
    """Gráfico de distribuição dos preços"""
    print(f"\n4. Gerando gráfico de distribuição para {coluna}...")
    
    try:
        # Verificar se o DataFrame tem MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # Para DataFrames com MultiIndex, tentar acessar a coluna correta
            if (coluna, 'DIS') in df.columns:
                dados = df[(coluna, 'DIS')].copy()
            else:
                # Tentar encontrar a coluna em qualquer nível
                col_disponiveis = [c for c in df.columns.get_level_values(0) if str(coluna) in str(c)]
                if not col_disponiveis:
                    print(f"Coluna {coluna} não encontrada no DataFrame.")
                    return None
                dados = df[col_disponiveis[0]].copy()
        else:
            # Para DataFrames sem MultiIndex
            if coluna not in df.columns:
                print(f"Coluna {coluna} não encontrada no DataFrame.")
                return None
            dados = df[coluna].copy()
        
        # Converter para numérico, forçando erros para NaN
        dados = pd.to_numeric(dados, errors='coerce').dropna()
        
        if dados.empty:
            print(f"Não há dados numéricos válidos para a coluna {coluna}.")
            return None
        
        plt.figure(figsize=(12, 6))
        sns.histplot(dados, kde=True, bins=50, color='green')
        plt.title(f'Distribuição do {coluna}', fontsize=16)
        plt.xlabel('Preço (USD)', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.grid(True)
        
        # Nome do arquivo sem caracteres inválidos
        nome_arquivo = f'distribuicao_{coluna.lower().replace(" ", "_")}'
        nome_arquivo = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in nome_arquivo)
        
        # Caminho completo do arquivo
        caminho_grafico = f'imagens/analise_exploratoria/{nome_arquivo}.png'
        
        plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo em: {caminho_grafico}")
        return caminho_grafico
        
    except Exception as e:
        print(f"Erro ao gerar o gráfico de distribuição: {e}")
        return None

def gerar_relatorio_completo():
    """Função principal que executa toda a análise"""
    # Verifica se o arquivo de dados já existe
    caminho_dados = 'dados/historico_DIS_20250727.csv'
    
    if os.path.exists(caminho_dados):
        print(f"Carregando dados de {caminho_dados}")
        
        try:
            # Ler o arquivo CSV, pulando as duas primeiras linhas (cabeçalho e linha de ticker)
            df = obter_dados()
            
            # Converter todas as colunas para numérico
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        except Exception as e:
            print(f"Erro ao processar o arquivo CSV: {e}")
            print("Tentando baixar os dados novamente...")
            df = obter_dados()
    else:
        print("Arquivo de dados não encontrado. Baixando dados...")
        df = obter_dados()
    
    # Ordenando o índice por data
    df = df.sort_index()
    
    # Remover linhas com todos os valores ausentes
    df = df.dropna(how='all')
    
    # Remover colunas que não têm dados
    df = df.dropna(axis=1, how='all')
    
    print("\n" + "="*50)
    print("ANÁLISE EXPLORATÓRIA DE DADOS")
    print("="*50)
    
    # 1. Verificar dados ausentes
    tabela_ausentes = verificar_dados_ausentes(df)
    
    # 2. Gerar estatísticas descritivas
    estatisticas = gerar_estatisticas_descritivas(df)
    
    # 3. Plotar série temporal (apenas se tivermos dados suficientes)
    if not df.empty and 'Close' in df.columns:
        plotar_serie_temporal(df)
    
    # 4. Plotar distribuição dos preços
    if not df.empty and 'Close' in df.columns:
        plotar_distribuicao(df, 'Close')
    
    print("\nAnálise exploratória concluída com sucesso!")
    
    # Retornando os dados para possível uso posterior
    return df, tabela_ausentes, estatisticas

if __name__ == "__main__":
    gerar_relatorio_completo()
