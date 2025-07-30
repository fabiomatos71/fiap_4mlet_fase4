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
DEFAULT_BATCH_SIZE = 32  # Melhor valor encontrado: 32
DEFAULT_EPOCHS = 50     # Melhor valor encontrado: 50
DEFAULT_LEARNING_RATE = 0.01  # Melhor valor encontrado: 0.01
DEFAULT_L2_LAMBDA = 0.0001    # Melhor valor encontrado: 0.0001
DEFAULT_DROPOUT_RATE = 0.1    # Melhor valor encontrado: 0.1
DEFAULT_LSTM_UNITS = (50, 25) # Melhor arquitetura encontrada
DEFAULT_DENSE_UNITS = 10      # Melhor valor encontrado: 10
DEFAULT_OPTIMIZER = 'adam'    # Melhor otimizador encontrado
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
    l2_lambda: float = DEFAULT_L2_LAMBDA,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    lstm_units: tuple = DEFAULT_LSTM_UNITS,
    dense_units: int = DEFAULT_DENSE_UNITS,
    optimizer_name: str = DEFAULT_OPTIMIZER
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
    model = Sequential()
    
    # Adiciona a camada de entrada explicitamente
    model.add(tf.keras.layers.Input(shape=(window_size, 1)))
    
    # Adiciona as camadas LSTM dinamicamente com base em lstm_units
    for i, units in enumerate(lstm_units, 1):
        model.add(LSTM(
            units=units,
            return_sequences=i < len(lstm_units),  # Retorna sequência se não for a última camada
            kernel_regularizer=l2(l2_lambda),
            name=f'lstm_{i}'
        ))
        model.add(Dropout(dropout_rate, name=f'dropout_{i}'))
    
    # Adiciona camada densa se especificado
    if dense_units > 0:
        model.add(Dense(
            dense_units,
            activation='relu',
            kernel_regularizer=l2(l2_lambda),
            name='dense_1'
        ))
        model.add(Dropout(dropout_rate/2, name='dropout_dense'))
        
        # Camada de saída
        model.add(Dense(1, activation='linear', name='output'))
    
    # Configurar o otimizador
    if optimizer_name.lower() == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Otimizador não suportado: {optimizer_name}")
    
    # Compilar o modelo
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
    l2_lambda: float = DEFAULT_L2_LAMBDA,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    lstm_units: tuple = DEFAULT_LSTM_UNITS,
    dense_units: int = DEFAULT_DENSE_UNITS,
    optimizer_name: str = DEFAULT_OPTIMIZER,
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
    
    # Criar e compilar o modelo com os parâmetros fornecidos
    model = criar_modelo(
        window_size=window_size,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda,
        dropout_rate=dropout_rate,
        lstm_units=lstm_units,
        dense_units=dense_units,
        optimizer_name=optimizer_name
    )
    
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
    # Hiperparâmetros do modelo
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f'Taxa de aprendizado (padrão: {DEFAULT_LEARNING_RATE})'
    )
    parser.add_argument(
        '--l2_lambda',
        type=float,
        default=DEFAULT_L2_LAMBDA,
        help=f'Fator de regularização L2 (padrão: {DEFAULT_L2_LAMBDA})'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help=f'Taxa de dropout (padrão: {DEFAULT_DROPOUT_RATE})'
    )
    parser.add_argument(
        '--lstm_units',
        type=str,
        default=','.join(map(str, DEFAULT_LSTM_UNITS)),
        help=f'Unidades nas camadas LSTM, separadas por vírgula (padrão: {DEFAULT_LSTM_UNITS})'
    )
    parser.add_argument(
        '--dense_units',
        type=int,
        default=DEFAULT_DENSE_UNITS,
        help=f'Unidades na camada densa (padrão: {DEFAULT_DENSE_UNITS})'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default=DEFAULT_OPTIMIZER,
        help=f'Otimizador a ser usado (adam, rmsprop, sgd) (padrão: {DEFAULT_OPTIMIZER})'
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
        
        # Processar argumentos
        lstm_units = tuple(map(int, args.lstm_units.split(',')))
        
        # 2. Treinar modelo com os hiperparâmetros otimizados
        print("\nIniciando treinamento do modelo...")
        print("\nParâmetros do modelo:")
        print(f"- Batch size: {args.batch_size}")
        print(f"- Épocas: {args.epochs}")
        print(f"- Taxa de aprendizado: {args.learning_rate}")
        print(f"- L2 lambda: {args.l2_lambda}")
        print(f"- Dropout rate: {args.dropout_rate}")
        print(f"- LSTM units: {lstm_units}")
        print(f"- Dense units: {args.dense_units}")
        print(f"- Otimizador: {args.optimizer}")
        
        model, history = treinar_modelo(
            dados=dados,
            window_size=window_size,
            batch_size=min(args.batch_size, len(dados['X_train'])),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2_lambda=args.l2_lambda,
            dropout_rate=args.dropout_rate,
            lstm_units=lstm_units,
            dense_units=args.dense_units,
            optimizer_name=args.optimizer,
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
