"""Métricas personalizadas para o modelo LSTM."""
import tensorflow as tf
import tensorflow.keras.backend as K

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return 100.0 * K.mean(diff, axis=-1)
