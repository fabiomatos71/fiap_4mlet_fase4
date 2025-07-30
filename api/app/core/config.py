from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Configurações da API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "API para predição de preços da ação da Disney"
    
    # Caminhos dos modelos
    MODEL_DIR: Path = Path("app/models")
    MODEL_FILE: str = "modelo_janela20_hiper.h5"
    SCALER_FILE: str = "escalonador_Close.joblib"
    
    # Configurações do modelo
    MODEL_VERSION: str = "1.0.0"
    
    class Config:
        case_sensitive = True

settings = Settings()
