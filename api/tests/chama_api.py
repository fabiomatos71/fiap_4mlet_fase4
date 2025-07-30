import requests
import json
import random
from typing import Dict, List, Any
import argparse

# URL base da API (ajuste conforme necessário)
BASE_URL = "http://localhost:8000/api/v1"

def generate_test_data(window_size: int = 20) -> Dict[str, List[float]]:
    """
    Gera dados de teste aleatórios para preços de fechamento.
    
    Args:
        window_size: Número de dias de dados históricos a serem gerados
        
    Returns:
        Dicionário no formato esperado pela API
    """
    # Gera preços iniciais entre 100 e 200
    base_price = random.uniform(100, 200)
    
    # Gera uma série temporal com alguma variação aleatória
    close_prices = [base_price]
    for _ in range(1, window_size):
        # Adiciona uma variação aleatória ao preço anterior
        variation = random.uniform(-2.0, 2.0)
        new_price = close_prices[-1] + variation
        close_prices.append(round(new_price, 2))
    
    return {"close": close_prices}

def test_predict(data: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Testa o endpoint de previsão da API.
    
    Args:
        data: Dados de entrada no formato {'close': [lista de preços]}
        
    Returns:
        Resposta da API
    """
    url = f"{BASE_URL}/predict"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao chamar a API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Resposta do servidor: {e.response.text}")
        return {}

def test_model_info() -> Dict[str, Any]:
    """
    Testa o endpoint de informações do modelo.
    
    Returns:
        Informações sobre o modelo
    """
    url = f"{BASE_URL}/model-info"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao obter informações do modelo: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Teste da API de previsão de ações')
    parser.add_argument('--dados', type=str, help='Arquivo JSON com dados de entrada (opcional)')
    parser.add_argument('--tamanho', type=int, default=20, help='Número de dias de dados históricos (padrão: 20)')
    
    args = parser.parse_args()
    
    # Testa o endpoint de informações do modelo
    print("\n=== Testando informações do modelo ===")
    model_info = test_model_info()
    if model_info:
        print("Informações do modelo:")
        print(json.dumps(model_info, indent=2, ensure_ascii=False))
    
    # Prepara os dados de teste
    if args.dados:
        # Carrega dados de um arquivo JSON
        try:
            with open(args.dados, 'r') as f:
                test_data = json.load(f)
            print(f"\nDados carregados do arquivo: {args.dados}")
        except Exception as e:
            print(f"Erro ao carregar arquivo {args.dados}: {e}")
            return
    else:
        # Gera dados de teste aleatórios
        test_data = generate_test_data(args.tamanho)
        print("\nDados de teste gerados aleatoriamente:")
        print(f"Número de dias: {len(test_data['close'])}")
        print(f"Primeiros 5 valores: {test_data['close'][:5]}...")
        print(f"Últimos 5 valores: ...{test_data['close'][-5:]}")
    
    # Testa o endpoint de previsão
    print("\n=== Testando previsão ===")
    prediction = test_predict(test_data)
    
    if prediction:
        print("\nResultado da previsão:")
        print(json.dumps(prediction, indent=2, ensure_ascii=False))
    
    print("\nTeste concluído!")

if __name__ == "__main__":
    main()
