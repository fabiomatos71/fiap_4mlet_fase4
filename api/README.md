# API de Previsão de Ações

API RESTful para prever preços de ações usando modelos LSTM treinados.

## Estrutura do Projeto

```
api/
├── app/                  # Código da aplicação
│   ├── __init__.py
│   ├── main.py           # Ponto de entrada da API
│   ├── models/           # Modelos Pydantic
│   │   └── schemas.py
│   ├── api/              # Rotas da API
│   │   └── endpoints/
│   │       └── predict.py
│   └── core/             # Lógica principal
│       ├── config.py
│       └── model_loader.py
├── models/               # Modelos treinados e scalers
├── tests/                # Testes automatizados
├── requirements.txt      # Dependências
└── run.py               # Script para iniciar a API
```

## Configuração

1. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   .\venv\Scripts\activate  # Windows
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Coloque os arquivos do modelo treinado e do scaler em `api/models/`:
   - `lstm_model.h5`
   - `scaler.pkl`

## Executando a API

```bash
# Modo desenvolvimento (com recarregamento automático)
python run.py
```

A API estará disponível em `http://localhost:8000`

## Documentação da API

- Documentação interativa: `http://localhost:8000/docs`
- Documentação alternativa: `http://localhost:8000/redoc`

## Endpoints

### Prever preço

```
POST /api/v1/predict
```

**Exemplo de requisição:**
```json
{
  "open": [100.0, 101.5, 102.3],
  "high": [101.0, 102.0, 103.0],
  "low": [99.5, 100.5, 101.5],
  "close": [100.5, 101.8, 102.5],
  "volume": [1000000, 1200000, 1500000]
}
```

**Resposta de sucesso:**
```json
{
  "predicted_price": 103.2,
  "confidence": 0.85,
  "model_version": "1.0.0"
}
```

### Informações do modelo

```
GET /api/v1/model-info
```

**Resposta de sucesso:**
```json
{
  "model_version": "1.0.0",
  "input_shape": [60, 5],
  "last_trained": "2023-01-01",
  "metrics": {
    "mae": 0.05,
    "mse": 0.01
  }
}
```

## Testes

Para executar os testes:

```bash
pytest tests/
```

## Implantação

Para produção, considere usar um servidor ASGI como Uvicorn com Gunicorn:

```bash
pip install gunicorn

gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```
