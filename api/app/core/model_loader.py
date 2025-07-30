import os
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from .config import settings

# Cache para o modelo e scaler
_model = None
_scaler = None

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    import tensorflow.keras.backend as K
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return 100.0 * K.mean(diff, axis=-1)

def obter_modelo() -> Tuple[tf.keras.Model, str]:
    """
    Retorna o modelo carregado em memória ou o carrega se necessário.
    
    Returns:
        - Modelo Keras carregado e compilado com as métricas apropriadas
        - Versão do modelo
        
    """
    model_version = "janela20_passo5_hiper_20250727"
    global _model
    if _model is None:
        model_path = settings.MODEL_DIR / settings.MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        
        print(f"Carregando modelo de: {model_path}")
        
        try:
            # Tenta carregar o modelo sem compilar primeiro
            _model = tf.keras.models.load_model(
                model_path,
                compile=False
            )
            print("Modelo carregado com sucesso (sem compilar)")
            
            # Verifica se o modelo foi carregado corretamente
            if _model is None:
                raise ValueError("Falha ao carregar o modelo: modelo é None")
                
            # Configuração básica de compilação
            optimizer = 'adam'
            loss = 'mse'
            
            # Tenta manter as configurações originais, se disponíveis
            if hasattr(_model, 'optimizer'):
                optimizer = _model.optimizer
            if hasattr(_model, 'loss'):
                loss = _model.loss
            
            # Compila o modelo com as métricas necessárias
            _model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=['mae', mape]  # Métricas usadas no treinamento
            )
            print("Modelo compilado com sucesso")
            
        except Exception as e:
            error_msg = f"Erro ao carregar o modelo: {str(e)}"
            print(error_msg)
            _model = None  # Garante que _model seja None em caso de erro
            raise ValueError(error_msg)
            
    return _model, model_version 

def obter_scaler():
    """Retorna o escalonador carregado em memória ou o carrega se necessário."""
    global _scaler
    if _scaler is None:
        scaler_path = settings.MODEL_DIR / settings.SCALER_FILE
        if not scaler_path.exists():
            raise FileNotFoundError(f"Escalonador não encontrado em {scaler_path}")
        _scaler = joblib.load(scaler_path)
    return _scaler

def prepara_entrada_predicao(data: dict, scaler) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara os dados para predição, seguindo o mesmo pré-processamento do treinamento.
    
    O modelo espera como entrada uma sequência temporal de preços de fechamento normalizados.
    
    Args:
        data: Dicionário contendo os dados de entrada com as chaves:
            - close: Lista de preços de fechamento
            - (outras features são ignoradas, pois o modelo foi treinado apenas com close)
        scaler: Scaler treinado para normalização dos dados
        
    Returns:
        Tuple contendo:
            - array 3D no formato (n_amostras, window_size, n_features)
            - None (placehoder para datas, não utilizado no momento)
    """
    try:
        # 1. Extrair apenas os preços de fechamento
        close_prices = np.array(data['close'], dtype=np.float32)
        
        # 2. Redimensionar para 2D (necessário para o scaler)
        close_prices_2d = close_prices.reshape(-1, 1)
        
        # 3. Suprimir avisos de feature names
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, 
                                 message="X does not have valid feature names")
            # 4. Aplicar a mesma normalização usada no treinamento
            close_prices_scaled = scaler.transform(close_prices_2d)
        
        # 5. Remover a dimensionalidade extra (volta para 1D)
        close_prices_scaled = close_prices_scaled.flatten()
        
        # 5. Criar sequências temporais no formato esperado pelo modelo
        # O modelo foi treinado com janelas de 20 passos temporais
        window_size = 20  # Mesmo valor usado no treinamento
        sequences = []
        
        # Criar sequências deslizantes
        for i in range(len(close_prices_scaled) - window_size + 1):
            sequences.append(close_prices_scaled[i:(i + window_size)])
        
        # Converter para array numpy
        X = np.array(sequences)
        
        # Adicionar dimensão de features (necessário para LSTM: [samples, timesteps, features])
        X = np.expand_dims(X, axis=2)
        
        return X, None
        
    except Exception as e:
        raise ValueError(f"Erro ao preparar os dados para predição: {str(e)}")
