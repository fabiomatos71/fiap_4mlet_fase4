from pydantic import BaseModel, Field
from typing import List

class StockData(BaseModel):
    """Schema para os dados de entrada da previsão"""
    close: List[float] = Field(..., description="Lista de preços de fechamento")

class PredictionResult(BaseModel):
    """Schema para o resultado da previsão"""
    predicted_price: float = Field(..., description="Preço previsto")
    confidence: float = Field(..., ge=0, le=1, description="Nível de confiança da previsão (0 a 1)")
    model_version: str = Field(..., description="Versão do modelo utilizado")

class ModelInfo(BaseModel):
    """Informações sobre o modelo carregado"""
    model_version: str = Field(..., description="Versão do modelo")
    input_shape: str = Field(..., description="Formato da entrada do modelo como string")
    last_trained: str = Field(..., description="Data do último treinamento no formato YYYY-MM-DD")
