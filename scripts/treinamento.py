#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análise e Treinamento do Modelo LSTM para Previsão de Ações da Disney (DIS)

Este script implementa o pipeline completo de treinamento de um modelo LSTM para
prever o preço de fechamento das ações da Disney com base em dados históricos.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from typing import Tuple, Dict, Optional
import json
from datetime import datetime

# Configuração de seeds para reprodutibilidade
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Constantes
DEFAULT_WINDOW_SIZE = 60
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.2


def carregar_dados(caminho_arquivo: str) -> dict:
    """
    Carrega e prepara os dados pré-processados do arquivo NPZ.
    
    Args:
        caminho_arquivo: Caminho para o arquivo NPZ com os dados processados
        
    Returns:
        Dicionário com os arrays de treino, validação e teste no formato 3D
    """
    try:
        # Carrega os dados
        dados = np.load(caminho_arquivo, allow_pickle=True)
        
        # Obtém o tamanho da janela a partir do formato dos dados
        window_size = dados['X_train'].shape[1] if len(dados['X_train'].shape) > 1 else 1
        
        # Função para redimensionar os dados para 3D se necessário
        def preparar_X(X):
            if len(X.shape) == 2:
                return X.reshape((X.shape[0], X.shape[1], 1))
            return X
            
        # Prepara os dados
        dados_preparados = {
            'X_train': preparar_X(dados['X_train']),# De 2D para 3D
            'y_train': dados['y_train'],
            'X_val': preparar_X(dados['X_val']),# De 2D para 3D
            'y_val': dados['y_val'],
            'X_test': preparar_X(dados['X_test']),# De 2D para 3D
            'y_test': dados['y_test'],
            'window_size': window_size
        }
        
        print("\nDados carregados com sucesso!")
        print(f"- Janela temporal: {window_size} dias")
        print(f"- Amostras de treino: {len(dados_preparados['X_train'])}")
        print(f"- Amostras de validação: {len(dados_preparados['X_val'])}")
        print(f"- Amostras de teste: {len(dados_preparados['X_test'])}")
        
        return dados_preparados
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        raise


def criar_modelo(
    window_size: int = DEFAULT_WINDOW_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2_lambda: float = 0.01,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Cria o modelo LSTM com a arquitetura especificada.
    
    Args:
        window_size: Tamanho da janela temporal
        learning_rate: Taxa de aprendizado inicial
        l2_lambda: Fator de regularização L2
        dropout_rate: Taxa de dropout
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        # Primeira camada LSTM
        LSTM(
            50,
            input_shape=(window_size, 1),
            return_sequences=True,
            kernel_regularizer=l2(l2_lambda),
            name='lstm_1'
        ),
        Dropout(dropout_rate, name='dropout_1'),
        
        # Segunda camada LSTM
        LSTM(
            25,
            return_sequences=False,
            kernel_regularizer=l2(l2_lambda),
            name='lstm_2'
        ),
        
        # Camada densa
        Dense(
            10,
            activation='relu',
            kernel_regularizer=l2(l2_lambda),
            name='dense_1'
        ),
        
        # Camada de saída
        Dense(1, activation='linear', name='output')
    ])
    
    # Compilar o modelo
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model


def configurar_callbacks(
    monitor: str = 'val_loss',
    patience_early_stop: int = 15,
    patience_reduce_lr: int = 5,
    factor_reduce_lr: float = 0.5,
    min_lr: float = 1e-6,
    log_dir: str = 'logs',
    model_checkpoint_path: str = 'melhor_modelo.h5'
) -> list:
    """
    Configura os callbacks para o treinamento.
    
    Args:
        monitor: Métrica a ser monitorada
        patience_early_stop: Paciência para early stopping
        patience_reduce_lr: Paciência para redução de LR
        factor_reduce_lr: Fator de redução do LR
        min_lr: LR mínimo
        log_dir: Diretório para logs do TensorBoard
        model_checkpoint_path: Caminho para salvar o melhor modelo
        
    Returns:
        Lista de callbacks configurados
    """
    callbacks = [
        # Early Stopping
        EarlyStopping(
            monitor=monitor,
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Redução de LR em platô
        ReduceLROnPlateau(
            monitor=monitor,
            factor=factor_reduce_lr,
            patience=patience_reduce_lr,
            min_lr=min_lr,
            verbose=1
        ),
        
        # Salvar o melhor modelo
        ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    return callbacks


def treinar_modelo(
    dados: Dict[str, np.ndarray],
    window_size: int = DEFAULT_WINDOW_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    output_dir: str = 'resultados',
    nome_modelo: str = 'melhor_modelo.h5'
) -> Tuple[tf.keras.Model, Dict]:
    """
    Treina o modelo LSTM com os dados fornecidos.
    
    Args:
        dados: Dicionário com os dados de treino, validação e teste
        window_size: Tamanho da janela temporal
        batch_size: Tamanho do lote
        epochs: Número máximo de épocas
        learning_rate: Taxa de aprendizado inicial
        output_dir: Diretório para salvar os resultados
        nome_modelo: Nome do arquivo para salvar o melhor modelo
        
    Returns:
        Tupla contendo o modelo treinado e o histórico de treinamento
    """
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar e compilar o modelo
    model = criar_modelo(window_size=window_size, learning_rate=learning_rate)
    
    # Configurar callbacks
    callbacks = configurar_callbacks(
        log_dir=os.path.join(output_dir, 'logs'),
        model_checkpoint_path=os.path.join(output_dir, nome_modelo)
    )
    
    # Treinar o modelo
    history = model.fit(
        dados['X_train'],
        dados['y_train'],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(dados['X_val'], dados['y_val']),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history.history


def avaliar_modelo(
    model: tf.keras.Model,
    dados: Dict[str, np.ndarray],
    output_dir: str = 'resultados'
) -> Dict[str, float]:
    """
    Avalia o modelo nos conjuntos de treino, validação e teste.
    
    Args:
        model: Modelo treinado
        dados: Dicionário com os dados
        output_dir: Diretório para salvar as métricas
        
    Returns:
        Dicionário com as métricas de avaliação
    """
    resultados = {}
    
    for conjunto in ['train', 'val', 'test']:
        X = dados[f'X_{conjunto}']
        y = dados[f'y_{conjunto}']
        
        if X.size == 0 or y.size == 0:
            continue
            
        # Fazer previsões
        y_pred = model.predict(X, verbose=0).flatten()
        y_true = y.flatten()
        
        # Calcular métricas
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Armazenar resultados
        resultados[f'{conjunto}_mse'] = float(mse)
        resultados[f'{conjunto}_mae'] = float(mae)
        resultados[f'{conjunto}_mape'] = float(mape)
        resultados[f'{conjunto}_r2'] = float(r2)
    
    # Salvar métricas em arquivo
    with open(os.path.join(output_dir, 'metricas.json'), 'w') as f:
        json.dump(resultados, f, indent=4)
    
    return resultados


def plot_resultados(
    history: Dict[str, list],
    metric: str = 'loss',
    output_dir: str = 'modelos'
) -> None:
    """
    Plota o histórico de treinamento.
    
    Args:
        history: Histórico de treinamento
        metric: Métrica a ser plotada
        output_dir: Diretório para salvar os gráficos
    """
    plt.figure(figsize=(10, 6))
    
    # Plotar métrica de treino
    if metric in history:
        plt.plot(history[metric], label=f'Treinamento {metric}')
    
    # Plotar métrica de validação
    val_metric = f'val_{metric}'
    if val_metric in history:
        plt.plot(history[val_metric], label=f'Validação {metric}')
    
    plt.title(f'Evolução da {metric} durante o treinamento')
    plt.xlabel('Época')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    
    # Salvar figura
    plt.savefig(os.path.join(output_dir, f'{metric}_evolution.png'))
    plt.close()


def main():
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(
        description='Treinar modelo LSTM para prever preços da Disney (DIS)'
    )
    
    # Argumentos obrigatórios
    parser.add_argument(
        'dados_entrada',
        type=str,
        help='Caminho para o arquivo NPZ com os dados processados (ex: dados/preparados/dados_preparados_janela60_passo5.npz)'
    )
    
    # Argumentos opcionais
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Tamanho do lote (padrão: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Número de épocas (padrão: {DEFAULT_EPOCHS})'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f'Taxa de aprendizado (padrão: {DEFAULT_LEARNING_RATE})'
    )
    parser.add_argument(
        '--nome_modelo',
        type=str,
        default='melhor_modelo.h5',
        help='Nome do arquivo para salvar o melhor modelo (padrão: melhor_modelo.h5)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='modelos',
        help='Diretório para salvar os resultados (padrão: modelos/)'
    )
    
    # Parse dos argumentos
    args = parser.parse_args()
    
    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 1. Carregar e preparar dados
        print(f"Carregando dados de {args.dados_entrada}...")
        dados = carregar_dados(args.dados_entrada)
        
        # Extrai o tamanho da janela dos dados carregados
        window_size = dados['window_size']
        
        # Verificar dimensões dos dados
        print("\nResumo dos dados carregados:")
        print(f"- Treino: {dados['X_train'].shape[0]} amostras, formato {dados['X_train'].shape}")
        print(f"- Validação: {dados['X_val'].shape[0]} amostras, formato {dados['X_val'].shape}")
        print(f"- Teste: {dados['X_test'].shape[0]} amostras, formato {dados['X_test'].shape}")
        print(f"- Tamanho da janela: {window_size} dias")
        
        # 2. Treinar modelo
        print("\nIniciando treinamento do modelo...")
        model, history = treinar_modelo(
            dados=dados,
            window_size=window_size,  # Usa o window_size extraído dos dados
            batch_size=min(args.batch_size, len(dados['X_train'])),  # Ajusta batch_size se necessário
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            nome_modelo=args.nome_modelo
        )
        
        # 3. Avaliar modelo
        print("\nAvaliando modelo...")
        metricas = avaliar_modelo(model, dados, output_dir=args.output_dir)
        
        # Exibir métricas
        print("\nMétricas de desempenho:")
        for nome, valor in metricas.items():
            print(f"- {nome}: {valor:.6f}")
        
        # 4. Salvar resultados
        print(f"\nResultados salvos em: {os.path.abspath(args.output_dir)}")
        
        # 5. Plotar métricas
        print("Gerando gráficos...")
        plot_resultados(history, 'loss', args.output_dir)
        plot_resultados(history, 'mae', args.output_dir)
        
        print("\nProcesso concluído com sucesso!")
        
    except Exception as e:
        print(f"\nErro durante a execução: {e}")
        raise


if __name__ == "__main__":
    main()
