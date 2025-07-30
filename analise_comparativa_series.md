# Análise Comparativa: Configurações de Sequência Temporal

## Introdução
Este documento apresenta uma análise comparativa entre três configurações de sequências temporais utilizadas no treinamento do modelo LSTM para previsão de preços de ações. As configurações testadas foram:

1. **Janela de 60 dias com passo de 5 dias** (com sobreposição de 55 dias)
2. **Janela de 60 dias com passo de 60 dias** (sem sobreposição)
3. **Janela de 20 dias com passo de 5 dias** (com sobreposição de 15 dias)

## Metodologia

### Configurações Testadas

#### Configuração 1: janela60_passo5
- **Tamanho da janela**: 60 dias
- **Passo**: 5 dias (sobreposição de 55 dias)
- **Número de amostras**: 239 sequências
- **Características**: Maior número de amostras devido à sobreposição

#### Configuração 2: janela60_passo60
- **Tamanho da janela**: 60 dias
- **Passo**: 60 dias (sem sobreposição)
- **Número de amostras**: 20 sequências
- **Características**: Amostras independentes, sem sobreposição

#### Configuração 3: janela20_passo5
- **Tamanho da janela**: 20 dias
- **Passo**: 5 dias (sobreposição de 15 dias)
- **Número de amostras**: 247 sequências
- **Características**: Janela menor com sobreposição, capturando padrões de curto prazo

### Métricas de Avaliação
- **MSE (Erro Quadrático Médio)**: Mede a média dos quadrados dos erros entre os valores previstos e reais.
- **MAE (Erro Absoluto Médio)**: Mede a média das diferenças absolutas entre os valores previstos e reais.
- **MAPE (Erro Percentual Absoluto Médio)**: Expressa o erro percentual médio em relação aos valores reais.
- **R² (Coeficiente de Determinação)**: Indica a proporção da variância da variável dependente que é previsível a partir das variáveis independentes.

## Resultados

### Métricas de Desempenho

| Métrica | janela60_passo5 (com sobreposição) | janela60_passo60 (sem sobreposição) | janela20_passo5 (curto prazo) |
|---------|-----------------------------------|------------------------------------|-----------------------------|
| **Treino** | | | |
| - MSE | 0,00342 | 0,00078 | 0,00401 |
| - MAE | 0,04623 | 0,02392 | 0,04984 |
| - MAPE | 63,96% | 21,51% | 56,42% |
| - R² | 0,9632 | 0,9919 | 0,9569 |
| **Validação** | | | |
| - MSE | 0,00459 | 0,00276 | 0,00335 |
| - MAE | 0,05754 | 0,04919 | 0,05083 |
| - MAPE | 27,61% | 21,46% | 25,86% |
| - R² | 0,0472 | -3,0181 | 0,3061 |
| **Teste** | | | |
| - MSE | 0,00534 | 0,00654 | 0,00415 |
| - MAE | 0,05475 | 0,05906 | 0,04806 |
| - MAPE | 32,32% | 31,41% | 29,37% |
| - R² | 0,3738 | -0,9855 | 0,5133 |

## Análise Comparativa

### Desempenho nos Dados de Treino
- A configuração `janela60_passo60` apresentou o melhor desempenho nos dados de treino, com R² de 0,9919 e o menor MSE (0,00078).
- A configuração `janela60_passo5` também apresentou bom desempenho, com R² de 0,9632.
- A configuração `janela20_passo5` mostrou desempenho competitivo (R² de 0,9569), indicando que a janela menor ainda captura padrões significativos.
- O MAPE mais alto foi observado na configuração com sobreposição de 60 dias (63,96%), enquanto a janela menor (20 dias) apresentou MAPE ligeiramente menor (56,42%).

### Desempenho nos Dados de Validação
- A configuração `janela20_passo5` apresentou o melhor equilíbrio entre treino e validação, com R² de 0,3061, significativamente melhor que as demais.
- A configuração `janela60_passo60` apresentou o menor MSE (0,00276) mas com R² negativo (-3,0181), indicando forte overfitting.
- A configuração `janela60_passo5` apresentou R² ligeiramente positivo (0,0472), mas ainda baixo, indicando que a sobreposição ajuda, mas não resolve completamente o problema de generalização.
- O MAPE na validação foi menor para a configuração `janela20_passo5` (25,86%), sugerindo melhor capacidade preditiva.

### Desempenho nos Dados de Teste
- A configuração `janela20_passo5` apresentou o melhor desempenho geral no conjunto de teste, com MSE de 0,00415 (22% menor que a segunda melhor configuração) e MAE de 0,04806 (12% menor que a segunda melhor).
- O MAPE mais baixo foi observado na configuração `janela20_passo5` (29,37%), seguido pela configuração sem sobreposição (31,41%) e pela configuração com sobreposição de 60 dias (32,32%).
- O R² mais alto foi observado na configuração `janela20_passo5` (0,5133), significativamente melhor que as demais configurações (0,3738 para janela60_passo5 e -0,9855 para janela60_passo60).

## Conclusões

1. A configuração `janela20_passo5` (janela de 20 dias, passo de 5 dias) emergiu como a mais equilibrada, apresentando:
   - Melhor R² em validação (0,3061) e teste (0,5133)
   - Menor MSE e MAE nos dados de teste
   - MAPE mais baixo entre as configurações testadas
   - Menor evidência de overfitting em comparação com as demais configurações

2. A configuração sem sobreposição (`janela60_passo60`) apresentou forte overfitting, com excelente desempenho nos dados de treino mas desempenho ruim na validação e teste (R² negativo).

3. A configuração com sobreposição de 60 dias (`janela60_passo5`) apresentou melhor capacidade de generalização que a sem sobreposição, mas ainda com limitações (R² de apenas 0,0472 na validação).

4. A janela de tempo menor (20 dias) mostrou-se mais eficaz para capturar padrões de curto prazo nos preços das ações, resultando em melhor generalização para dados não vistos.

5. As métricas de validação sugerem que o modelo com janela de 20 dias e passo de 5 dias é o mais adequado para previsões de curto prazo, equilibrando bem a capacidade de aprendizado com a capacidade de generalização.

