# Logistic Regression - Vectorized Implementation | Regressão Logística - Implementação Vetorizada

> **English version below** | **Versão em português abaixo**

---

## English

A from-scratch implementation of Logistic Regression using NumPy for binary classification on the Framingham Heart Study dataset.

### Overview

This notebook implements the core logistic regression algorithm manually, focusing on understanding the mathematical foundations of the algorithm rather than using pre-built ML libraries.

### Dataset

**Framingham Heart Study** - Cardiovascular disease prediction dataset
- 4,238 samples with 15 features (demographics, risk factors, health metrics)
- Binary classification: 10-year CHD risk prediction
- Train/Test Split: 80/20 (3,390 / 848 samples)

### Implementation Details

#### Implemented From Scratch

- **Sigmoid function**: Logistic activation function
- **Cost function**: Binary cross-entropy (log loss)
- **Gradient computation**: Fully vectorized gradient calculation
- **Gradient descent**: Iterative optimization algorithm
- **Prediction function**: Binary classification with configurable threshold

#### External Dependencies (Non-Algorithm)

The following sklearn utilities are used for **data preprocessing only**:
- `train_test_split`: Dataset splitting
- `SimpleImputer`: Missing value imputation
- `StandardScaler`: Feature normalization
- `confusion_matrix`, `classification_report`: Evaluation metrics

**Note**: These are infrastructure utilities and do not affect the core algorithm implementation, which is 100% from scratch.

### Mathematical Foundation

#### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```

#### Cost Function (Binary Cross-Entropy)
```
J(w,b) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
```

#### Gradient Descent Update
```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

Where:
- `∂J/∂w = 1/m * X^T * (h(X) - y)` (vectorized)
- `∂J/∂b = 1/m * Σ(h(x) - y)`

### Results

#### Model Performance

- **Test Accuracy**: 63.68%
- **Precision (Class 1)**: 0.24
- **Recall (Class 1)**: 0.66
- **F1-Score (Class 1)**: 0.35

#### Analysis

The model shows high recall but low precision for the positive class (CHD risk). This is expected given:
1. Highly imbalanced dataset (~15% positive cases)
2. Medical prediction task where missing true positives is costly

The relatively low accuracy reflects the challenge of predicting cardiovascular risk from limited features and class imbalance.

### Key Features

- **Fully vectorized operations** using NumPy for computational efficiency
- **Epsilon smoothing** (1e-5) to prevent log(0) errors
- **Modular design** with separate functions for cost, gradient, and prediction
- **Training history tracking** for convergence analysis

### Usage

```python
# Initialize parameters
initial_w = np.zeros(n_features)
initial_b = 0.0

# Train model
w, b, J_history, w_history = gradient_descent(
    X_train_scaled,
    y_train,
    initial_w,
    initial_b,
    compute_cost,
    compute_gradient,
    alpha=0.001,        # learning rate
    num_iters=1000,     # iterations
    lambda_=0           # regularization
)

# Make predictions
predictions = predict(X_test_scaled, w, b, threshold=0.15)

# Evaluate
accuracy = np.mean(predictions == y_test) * 100
```

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Educational Purpose

This implementation prioritizes **clarity and understanding** over production-ready performance. It demonstrates:
- How logistic regression works mathematically
- The power of vectorization in machine learning
- Gradient descent optimization from first principles
- Binary classification fundamentals

---

## Português

Uma implementação do zero de Regressão Logística usando NumPy para classificação binária no dataset Framingham Heart Study.

### Visão Geral

Este notebook implementa o algoritmo central de regressão logística manualmente, focando na compreensão dos fundamentos matemáticos do algoritmo ao invés de usar bibliotecas de ML prontas.

### Dataset

**Framingham Heart Study** - Dataset de predição de doenças cardiovasculares
- 4.238 amostras com 15 features (demografia, fatores de risco, métricas de saúde)
- Classificação binária: predição de risco cardiovascular em 10 anos
- Divisão Treino/Teste: 80/20 (3.390 / 848 amostras)

### Detalhes da Implementação

#### Implementado do Zero

- **Função sigmoid**: Função de ativação logística
- **Função de custo**: Entropia cruzada binária (log loss)
- **Cálculo do gradiente**: Cálculo vetorizado completo do gradiente
- **Gradiente descendente**: Algoritmo de otimização iterativa
- **Função de predição**: Classificação binária com threshold configurável

#### Dependências Externas (Não-Algorítmicas)

As seguintes utilidades do sklearn são usadas **apenas para pré-processamento de dados**:
- `train_test_split`: Divisão do dataset
- `SimpleImputer`: Imputação de valores faltantes
- `StandardScaler`: Normalização de features
- `confusion_matrix`, `classification_report`: Métricas de avaliação

**Nota**: Estas são utilidades de infraestrutura e não afetam a implementação central do algoritmo, que é 100% do zero.

### Fundamentos Matemáticos

#### Função Sigmoid
```
σ(z) = 1 / (1 + e^(-z))
```

#### Função de Custo (Entropia Cruzada Binária)
```
J(w,b) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
```

#### Atualização do Gradiente Descendente
```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

Onde:
- `∂J/∂w = 1/m * X^T * (h(X) - y)` (vetorizado)
- `∂J/∂b = 1/m * Σ(h(x) - y)`

### Resultados

#### Performance do Modelo

- **Acurácia no Teste**: 63.68%
- **Precision (Classe 1)**: 0.24
- **Recall (Classe 1)**: 0.66
- **F1-Score (Classe 1)**: 0.35

#### Análise

O modelo mostra alto recall mas baixa precisão para a classe positiva (risco cardiovascular). Isso é esperado dado:
1. Dataset altamente desbalanceado (~15% casos positivos)
2. Tarefa de predição médica onde perder positivos verdadeiros é custoso

A acurácia relativamente baixa reflete o desafio de prever risco cardiovascular com features limitadas e desbalanceamento de classes.

### Características Principais

- **Operações totalmente vetorizadas** usando NumPy para eficiência computacional
- **Suavização epsilon** (1e-5) para prevenir erros de log(0)
- **Design modular** com funções separadas para custo, gradiente e predição
- **Rastreamento do histórico de treinamento** para análise de convergência

### Uso

```python
# Inicializar parâmetros
initial_w = np.zeros(n_features)
initial_b = 0.0

# Treinar modelo
w, b, J_history, w_history = gradient_descent(
    X_train_scaled,
    y_train,
    initial_w,
    initial_b,
    compute_cost,
    compute_gradient,
    alpha=0.001,        # taxa de aprendizado
    num_iters=1000,     # iterações
    lambda_=0           # regularização
)

# Fazer predições
predictions = predict(X_test_scaled, w, b, threshold=0.15)

# Avaliar
accuracy = np.mean(predictions == y_test) * 100
```

### Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Propósito Educacional

Esta implementação prioriza **clareza e compreensão** ao invés de performance pronta para produção. Ela demonstra:
- Como a regressão logística funciona matematicamente
- O poder da vetorização em machine learning
- Otimização via gradiente descendente desde os primeiros princípios
- Fundamentos de classificação binária

---

## License | Licença

Educational project - Free to use and modify.

Projeto educacional - Livre para usar e modificar.
