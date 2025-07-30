# Guia de Uso: Script de Treinamento e Busca de Hiperparâmetros

Este documento fornece instruções detalhadas sobre como usar o script `hyperparam_treinamento.py` para treinar modelos LSTM e realizar buscas de hiperparâmetros.

## Pré-requisitos

- Python 3.8+
- Pacotes Python listados em `requirements.txt`
- Dados de treinamento pré-processados no formato NPZ

## Instalação

1. Crie e ative um ambiente virtual (recomendado):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # ou
   .venv\Scripts\activate    # Windows
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso Básico

### Treinamento Padrão

Para treinar um modelo com parâmetros padrão:

```bash
python scripts/hyperparam_treinamento.py dados/preparados/dados_preparados_janela60_passo5.npz \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 0.001 \
  --output_dir modelos/janela60_passo5
```

### Busca de Hiperparâmetros

Para realizar uma busca por melhores hiperparâmetros:

```bash
python scripts/hyperparam_treinamento.py dados/preparados/dados_preparados_janela60_passo5.npz \
  --buscar_hiperparametros \
  --n_iter 10 \
  --cv_folds 3 \
  --epochs 50 \
  --output_dir modelos/busca_hiperparametros
```

## Parâmetros da Linha de Comando

### Parâmetros Obrigatórios

- `dados_entrada`: Caminho para o arquivo NPZ com os dados pré-processados

### Parâmetros Opcionais

- `--batch_size`: Tamanho do lote para treinamento (padrão: 32)
- `--epochs`: Número de épocas para treinamento (padrão: 100)
- `--learning_rate`: Taxa de aprendizado inicial (padrão: 0.001)
- `--output_dir`: Diretório para salvar os resultados (padrão: 'resultados')

### Parâmetros para Busca de Hiperparâmetros

- `--buscar_hiperparametros`: Ativa a busca por hiperparâmetros
- `--n_iter`: Número de iterações para busca aleatória (padrão: 10)
- `--cv_folds`: Número de folds para validação cruzada (padrão: 3)

## Espaço de Busca de Hiperparâmetros

O script busca automaticamente nos seguintes intervalos:

- Taxa de aprendizado: [0.1, 0.01, 0.001, 0.0001]
- Fator de regularização L2: [0.1, 0.01, 0.001, 0.0001]
- Taxa de dropout: [0.1, 0.2, 0.3, 0.4, 0.5]
- Unidades LSTM: [(64, 32), (128, 64), (64, 32, 16), (128, 64, 32)]
- Unidades densas: [0, 10, 20, 50]
- Tamanho do lote: [16, 32, 64, 128]
- Otimizadores: ['adam', 'rmsprop', 'sgd']
- Épocas: [50, 100, 150]

## Saída

### Durante o Treinamento

- Logs de progresso no console
- Gráficos de perda e métricas
- Modelo salvo em formato HDF5

### Após a Busca de Hiperparâmetros

- `melhores_parametros.json`: Melhores parâmetros encontrados
- `resultados_busca.json`: Resultados completos da busca
- `modelo_final.h5`: Modelo treinado com os melhores parâmetros
- Gráficos no diretório especificado

## Exemplos Avançados

### Usando Menos Iterações para Teste Rápido

```bash
python scripts/hyperparam_treinamento.py dados/preparados/dados_preparados_janela60_passo5.npz \
  --buscar_hiperparametros \
  --n_iter 5 \
  --cv_folds 2 \
  --epochs 30 \
  --output_dir resultados/teste_rapido
```

### Treinando com os Melhores Parâmetros Encontrados

Após encontrar os melhores parâmetros, você pode treinar por mais épocas:

```bash
python scripts/hyperparam_treinamento.py dados/preparados/dados_preparados_janela60_passo5.npz \
  --batch_size 128 \
  --epochs 200 \
  --learning_rate 0.0001 \
  --output_dir resultados/treinamento_final
```

## Dicas

1. Comece com um número pequeno de iterações para testar
2. Aumente o número de épocas para o treinamento final
3. Monitore o uso de memória durante a busca
4. Verifique os logs para identificar possíveis problemas
5. Considere ajustar o espaço de busca com base nos resultados iniciais
