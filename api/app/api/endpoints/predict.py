from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import numpy as np
from ...core.model_loader import obter_modelo, obter_scaler, prepara_entrada_predicao
from ...models.schemas import StockData, PredictionResult, ModelInfo

router = APIRouter()

@router.post("/predict", response_model=PredictionResult)
async def predict_stock_price(data: StockData):
    """
    Faz a previsão do preço da ação com base nos dados históricos fornecidos.
    
    O endpoint espera uma série temporal de preços de fechamento (pelo menos 20 dias)
    e retorna a previsão para o próximo período.
    
    - **close**: Lista de preços de fechamento (obrigatório, pelo menos 60 dias)
    """
    try:
        # Verifica se há dados suficientes para previsão
        if len(data.close) < 20:
            raise HTTPException(
                status_code=400,
                detail="São necessários pelo menos 20 dias de dados históricos para fazer previsões"
            )
            
        model, model_version = obter_modelo()
        scaler = obter_scaler()
        
        # Prepara os dados no formato esperado pelo modelo
        X, _ = prepara_entrada_predicao(data.dict(), scaler)
        
        # Faz a previsão
        y_pred_scaled = model.predict(X)
        
        # Desfaz a normalização da previsão
        # Cria um array com zeros no formato esperado pelo scaler
        dummy_array = np.zeros((len(y_pred_scaled), 1))
        dummy_array[:, 0] = y_pred_scaled.flatten()
        y_pred = scaler.inverse_transform(dummy_array)[:, 0]
        
        # Retorna a última previsão (mais recente)
        predicted_price = float(y_pred[-1])
        
        # Calcula um valor de confiança baseado na variação das últimas previsões
        # Este é um valor simulado - em produção, você pode querer usar uma métrica mais sofisticada
        confidence = min(0.99, max(0.7, 1.0 - np.std(y_pred[-5:]) / (np.mean(y_pred[-5:]) + 1e-7)))
        
        return {
            "predicted_price": predicted_price,
            "confidence": float(confidence),
            "model_version": model_version
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e.detail))

@router.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """
    Retorna informações sobre o modelo carregado.
    
    Inclui detalhes como arquitetura, métricas e formato de entrada esperado.
    """
    try:
        model, model_version = obter_modelo()
        
        # Verifica se o modelo foi carregado corretamente
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Falha ao carregar o modelo"
            )
            
        # Obtém o input_shape do modelo, se disponível
        input_shape = "Desconhecido"
        if hasattr(model, 'input_shape') and model.input_shape is not None:
            # Remove a dimensão do batch (None) e converte para string
            input_shape = str(model.input_shape[1:])
            
        # Informações básicas do modelo
        model_info = {
            "model_version": model_version,
            "input_shape": input_shape,  # Já formatado anteriormente
            "last_trained": "2025-07-27"
        }
            
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e.detail))
