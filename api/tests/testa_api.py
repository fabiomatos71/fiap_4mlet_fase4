#!/usr/bin/env python3
"""
Script para testar a API de previsão de preços de ações.
Obtém os últimos 20 dias de dados da Disney (DIS) e faz uma previsão para o dia atual.
"""
import sys
import json
from datetime import datetime, timedelta
import yfinance as yf
import requests

def get_historical_data():
    """
    Obtém os dados históricos dos últimos 20 dias úteis da Disney (DIS).
    Retorna uma lista com os preços de fechamento.
    """
    try:
        # Calcula a data de ontem
        hoje = datetime.now()
        ontem = hoje - timedelta(days=1)
        data_inicio = ontem - timedelta(days=30)  # Pega 30 dias para garantir 20 dias úteis
        
        # Formata as datas para o formato aceito pelo yfinance
        data_inicio_str = data_inicio.strftime('%Y-%m-%d')
        data_fim_str = ontem.strftime('%Y-%m-%d')
        
        print(f"Obtendo dados de {data_inicio_str} até {data_fim_str}...")
        
        # Baixa os dados da Disney (DIS) com auto_adjust=False para garantir a estrutura esperada
        acao = yf.download('DIS', start=data_inicio_str, end=hoje.strftime('%Y-%m-%d'), auto_adjust=False)
        
        if acao.empty:
            print("Erro: Não foi possível obter os dados da ação.")
            return None
            
        print("\nEstrutura dos dados recebidos:")
        print(acao.head())
        print("\nColunas:", acao.columns)
        
        # Acessa corretamente a coluna Close no MultiIndex
        if ('Close', 'DIS') not in acao.columns:
            print("Erro: Coluna 'Close' para 'DIS' não encontrada nos dados.")
            print("Colunas disponíveis:", acao.columns.tolist())
            return None
            
        # Pega os últimos 20 dias úteis
        close_series = acao[('Close', 'DIS')].tail(20)
        
        if len(close_series) < 20:
            print(f"Erro: Apenas {len(close_series)} dias úteis encontrados. São necessários 20.")
            return None
        
        print(f"\nDados obtidos com sucesso! Período: {close_series.index[0].strftime('%Y-%m-%d')} a {close_series.index[-1].strftime('%Y-%m-%d')}")
        print("\nÚltimos 5 preços de fechamento:")
        print(close_series.tail())
        
        # Converte para lista de valores
        close_prices = close_series.tolist()
        print("\nLista de preços:", [f"${p:.2f}" for p in close_prices])
        
        return close_prices
        
    except Exception as e:
        print(f"Erro ao obter dados históricos: {str(e.detail)}")
        return None

def predict_price(close_prices):
    """
    Chama a API para fazer a previsão do preço.
    """
    url = "http://localhost:8000/api/v1/predict"
    
    data = {
        "close": close_prices
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro na chamada da API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Resposta do servidor: {e.response.text}")
        return None

def main():
    print("=== Teste da API de Previsão de Preços ===\n")
    
    # Obtém os dados históricos
    close_prices = get_historical_data()
    if not close_prices:
        sys.exit(1)
    
    print("\nÚltimos 20 preços de fechamento:")
    for i in range(len(close_prices)):
        price = close_prices[i]
        print(f"Dia -{len(close_prices)-i}: ${price:.2f}")
    
    # Faz a previsão
    print("\nFazendo previsão para hoje ...")
    resultado = predict_price(close_prices)
    
    if resultado:
        print("\n=== Resultado da Previsão ===")
        print(f"Data da previsão: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nPreço previsto para hoje: ${resultado['predicted_price']:.2f}")
        print(f"\nNível de confiança: {resultado['confidence']*100:.1f}%")
        print(f"Versão do modelo: {resultado['model_version']}")
    else:
        print("\nNão foi possível obter a previsão.")
        sys.exit(1)

if __name__ == "__main__":
    main()
