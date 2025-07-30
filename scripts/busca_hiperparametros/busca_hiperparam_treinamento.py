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
from functools import partial
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scikeras.wrappers import KerasRegressor
import joblib

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
    dropout_rate: float = 0.2,
    lstm_units: tuple = (50, 25),
    dense_units: int = 10,
    optimizer: str = 'adam'
) -> tf.keras.Model:
    """
    Cria o modelo LSTM com a arquitetura especificada.
    
    Args:
        window_size: Tamanho da janela temporal
        learning_rate: Taxa de aprendizado inicial
        l2_lambda: Fator de regularização L2
        dropout_rate: Taxa de dropout
        lstm_units: Tupla com o número de unidades para cada camada LSTM
        dense_units: Número de unidades na camada densa
        optimizer: Nome do otimizador a ser usado ('adam', 'rmsprop', 'sgd')
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential()
    
    # Adiciona a camada de entrada explicitamente
    model.add(tf.keras.layers.Input(shape=(window_size, 1)))
    
    # Adiciona a primeira camada LSTM
    model.add(LSTM(
        units=lstm_units[0],
        return_sequences=len(lstm_units) > 1,  # Retorna sequência se houver mais camadas
        kernel_regularizer=l2(l2_lambda),
        name='lstm_1'
    ))
    model.add(Dropout(dropout_rate, name='dropout_1'))
    
    # Adiciona camadas LSTM adicionais
    for i in range(1, len(lstm_units)):
        model.add(LSTM(
            units=lstm_units[i],
            return_sequences=i < len(lstm_units) - 1,  # Retorna sequência se não for a última camada
            kernel_regularizer=l2(l2_lambda),
            name=f'lstm_{i+1}'
        ))
        model.add(Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Adiciona camada densa
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
    
    # Configura o otimizador
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Otimizador não suportado: {optimizer}")
    
    # Compila o modelo
    model.compile(
        optimizer=opt,
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
    output_dir: str = 'resultados'
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
        model_checkpoint_path=os.path.join(output_dir, 'melhor_modelo.h5')
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


def buscar_melhores_hiperparametros(
    dados: Dict[str, np.ndarray],
    output_dir: str = 'resultados/hyperparam_search',
    n_iter: int = 20,
    cv: int = 3,
    verbose: int = 1,
    n_jobs: int = -1
) -> Dict:
    """
    Realiza uma busca aleatória por hiperparâmetros usando RandomizedSearchCV.
    
    Args:
        dados: Dicionário com os dados de treino e validação
        output_dir: Diretório para salvar os resultados
        n_iter: Número de iterações da busca aleatória
        cv: Número de folds para validação cruzada
        verbose: Nível de verbosidade
        n_jobs: Número de jobs para rodar em paralelo (-1 para usar todos os núcleos)
        
    Returns:
        Dicionário com os melhores parâmetros encontrados
    """
    # Cria o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Define o espaço de busca de hiperparâmetros
    param_dist = {
        'model__learning_rate': [0.1, 0.01, 0.001, 0.0001],
        'model__l2_lambda': [0.1, 0.01, 0.001, 0.0001],
        'model__dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'model__lstm_units': [(50, 25), (64, 32), (128, 64), (64, 32, 16), (128, 64, 32)],
        'model__dense_units': [0, 10, 20, 50],
        'batch_size': [16, 32, 64, 128],
        'optimizer': ['adam', 'rmsprop', 'sgd'],
        'epochs': [50, 100, 150]
    }
    
    # Cria o modelo KerasRegressor com scikeras
    model = KerasRegressor(
        model=partial(
            criar_modelo,
            window_size=dados['X_train'].shape[1]
        ),
        optimizer=Adam(),  # Otimizador padrão, pode ser sobrescrito
        loss='mse',
        metrics=['mae', 'mape'],
        verbose=0
    )
    
    # Configura a validação cruzada para séries temporais
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Cria o RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=SEED,
        return_train_score=True
    )
    
    print("\nIniciando busca por hiperparâmetros...")
    print(f"Número de combinações: {n_iter}")
    print(f"Número de folds: {cv}")
    print(f"Total de treinamentos: {n_iter * cv}")
    
    # Executa a busca
    X_train = np.concatenate([dados['X_train'], dados['X_val']])
    y_train = np.concatenate([dados['y_train'], dados['y_val']])
    
    random_search.fit(X_train, y_train)
    
    # Processa os resultados
    resultados = {
        'melhores_parametros': random_search.best_params_.copy(),
        'melhor_pontuacao': float(random_search.best_score_),
        'melhor_estimador': str(random_search.best_estimator_),
        'resultados': []
    }
    
    # Remove o prefixo 'model__' dos parâmetros para facilitar a leitura
    for key in list(resultados['melhores_parametros'].keys()):
        if key.startswith('model__'):
            resultados['melhores_parametros'][key[7:]] = resultados['melhores_parametros'].pop(key)
    
    # Adiciona todos os resultados
    for i in range(len(random_search.cv_results_['params'])):
        params = random_search.cv_results_['params'][i].copy()
        # Remove o prefixo 'model__' dos parâmetros
        clean_params = {}
        for key, value in params.items():
            if key.startswith('model__'):
                clean_params[key[7:]] = value
            else:
                clean_params[key] = value
                
        resultados['resultados'].append({
            'rank': i+1,
            'params': clean_params,
            'mean_score': float(random_search.cv_results_['mean_test_score'][i]),
            'std_score': float(random_search.cv_results_['std_test_score'][i]),
            'mean_train_score': float(random_search.cv_results_['mean_train_score'][i])
        })
    
    # Ordena os resultados por pontuação média (melhor primeiro)
    resultados['resultados'].sort(key=lambda x: x['mean_score'], reverse=True)
    
    # Salva os resultados em um arquivo JSON
    resultados_path = os.path.join(output_dir, 'resultados_busca.json')
    with open(resultados_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    # Salva os melhores parâmetros em um arquivo separado
    best_params_path = os.path.join(output_dir, 'melhores_parametros.json')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(resultados['melhores_parametros'], f, indent=2, ensure_ascii=False)
    
    print(f"\nBusca concluída! Melhor pontuação: {random_search.best_score_:.6f}")
    print(f"Melhores parâmetros: {resultados['melhores_parametros']}")
    print(f"Resultados salvos em: {resultados_path}")
    print(f"Melhores parâmetros salvos em: {best_params_path}")
    
    return resultados


def plot_resultados(
    history: Dict[str, list],
    metric: str = 'loss',
    output_dir: str = 'resultados'
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
    # Configuração do parser de argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Treinar modelo LSTM para previsão de ações')
    
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
        help=f'Tamanho do lote (batch size) para treinamento (padrão: {DEFAULT_BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Número de épocas para treinamento (padrão: {DEFAULT_EPOCHS})'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f'Taxa de aprendizado inicial (padrão: {DEFAULT_LEARNING_RATE})'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='resultados',
        help='Diretório para salvar os resultados (padrão: resultados/)'
    )
    
    # Argumento para busca de hiperparâmetros
    parser.add_argument(
        '--buscar_hiperparametros',
        action='store_true',
        help='Se especificado, realiza uma busca por melhores hiperparâmetros'
    )
    
    parser.add_argument(
        '--n_iter',
        type=int,
        default=10,
        help='Número de iterações para busca de hiperparâmetros (padrão: 10)'
    )
    
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=3,
        help='Número de folds para validação cruzada (padrão: 3)'
    )
    
    # Parse dos argumentos
    args = parser.parse_args()
    
    try:
        # Carregar os dados
        print(f"\nCarregando dados de {args.dados_entrada}...")
        dados = carregar_dados(args.dados_entrada)
        
        # Criar diretório de saída
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.buscar_hiperparametros:
            # Realizar busca de hiperparâmetros
            print("\nIniciando busca por melhores hiperparâmetros...")
            resultados = buscar_melhores_hiperparametros(
                dados=dados,
                output_dir=os.path.join(args.output_dir, 'hyperparam_search'),
                n_iter=args.n_iter,
                cv=args.cv_folds,
                verbose=1
            )
            
            # Treinar o modelo final com os melhores parâmetros encontrados
            print("\nTreinando modelo final com os melhores parâmetros...")
            melhores_params = resultados['melhores_parametros']
            
            # Extrair parâmetros específicos para o treinamento
            batch_size = melhores_params.pop('batch_size')
            epochs = melhores_params.pop('epochs')
            optimizer_type = melhores_params.pop('optimizer')
            learning_rate = melhores_params.pop('learning_rate')
            
            # Criar e treinar o modelo final
            model = criar_modelo(
                window_size=dados['X_train'].shape[1],
                learning_rate=learning_rate,
                optimizer=optimizer_type,
                **{k: v for k, v in melhores_params.items() if k in ['l2_lambda', 'dropout_rate', 'lstm_units', 'dense_units']}
            )
            
            # Configurar callbacks
            callbacks = configurar_callbacks(
                monitor='val_loss',
                model_checkpoint_path=os.path.join(args.output_dir, 'melhor_modelo.h5'),
                log_dir=os.path.join(args.output_dir, 'logs')
            )
            
            # Treinar o modelo
            history = model.fit(
                dados['X_train'],
                dados['y_train'],
                validation_data=(dados['X_val'], dados['y_val']),
                batch_size=batch_size,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Salvar o modelo final
            model_path = os.path.join(args.output_dir, 'modelo_final.h5')
            model.save(model_path)
            print(f"\nModelo final salvo em: {os.path.abspath(model_path)}")
            
        else:
            # Treinar o modelo com os parâmetros fornecidos
            print("\nIniciando treinamento do modelo...")
            model, history = treinar_modelo(
                dados=dados,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir
            )
        
        # Avaliar o modelo
        print("\nAvaliando modelo...")
        metricas = avaliar_modelo(model, dados, output_dir=args.output_dir)
        
        # Plotar resultados
        print("\nGerando gráficos...")
        # Verifica se history é um dicionário ou um objeto History do Keras
        history_dict = history.history if hasattr(history, 'history') else history
        plot_resultados(history_dict, output_dir=args.output_dir)
        
        print(f"\nProcesso concluído com sucesso!")
        print(f"Resultados salvos em: {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        print(f"\nErro durante a execução: {e}")
        raise


if __name__ == "__main__":
    main()
