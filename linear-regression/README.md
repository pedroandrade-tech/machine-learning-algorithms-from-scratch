# Linear Regression - From Scratch Implementations | Regressão Linear - Implementações do Zero

> **🇺🇸 English version below** | **🇧🇷 Versão em português abaixo**

---

## 🇺🇸 English

From-scratch implementations of Linear Regression using NumPy, demonstrating both single-feature and multiple-feature regression on the Diabetes dataset.

### Overview

These notebooks implement linear regression algorithms manually, focusing on understanding the mathematical foundations of gradient descent optimization for regression problems.

### Notebooks

1. **Linear_Regression_Single_Feature_From_Scratch.ipynb**
   - Simple linear regression with one feature (BMI)
   - Visual demonstration of cost function optimization
   - Step-by-step gradient descent

2. **Multiple_Linear_Regression_From_Scratch.ipynb**
   - Multiple linear regression with 10 features
   - Demonstrates handling multiple predictors
   - Feature scaling and convergence analysis

### Dataset

**Diabetes Dataset** (from sklearn.datasets)
- **442 samples** of diabetes patients
- **10 baseline features**: age, sex, BMI, blood pressure, and 6 blood serum measurements
- **Target**: Quantitative measure of disease progression one year after baseline

**Features used**:
- Single Feature: BMI (Body Mass Index) only
- Multiple Features: All 10 baseline variables

### Implementation Details

#### ✅ Implemented From Scratch

- **Cost function**: Mean Squared Error (MSE)
- **Gradient computation**: Partial derivatives calculation
- **Gradient descent**: Iterative optimization algorithm
- **Prediction function**: Linear model evaluation
- **Performance metrics**: Manual R² calculation

#### ⚠️ External Dependencies (Non-Algorithm)

The following sklearn utilities are used for **data loading and evaluation only**:
- `load_diabetes`: Dataset loading
- `r2_score`: R² metric for model comparison

**Note**: The core regression algorithm (cost, gradient, optimization) is 100% from scratch.

### Mathematical Foundation

#### Linear Model
```
f(x) = w·x + b
```
- Single feature: `f(x) = w*x + b`
- Multiple features: `f(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`

#### Cost Function (MSE)
```
J(w,b) = 1/(2m) * Σ(f(x) - y)²
```

#### Gradient Descent Update
```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

Where:
- `∂J/∂w = 1/m * Σ(f(x) - y) * x`
- `∂J/∂b = 1/m * Σ(f(x) - y)`

### Results

#### Single Feature Linear Regression

- **Final Cost (MSE)**: 2,531.50
- **RMSE**: 50.31 points
- **R²**: 0.1462 (14.62% variance explained)
- **Training**: 3,000 iterations with α=0.01

**Analysis**: Single feature (BMI) explains only ~15% of disease progression, showing the limitation of using a single predictor.

#### Multiple Linear Regression

- **Final Cost (MSE)**: 1,452.19
- **R²**: 0.5102 (51.02% variance explained)
- **Training**: 10,000 iterations with α=0.01
- **Features**: 10 baseline variables

**Analysis**: Using all 10 features improves prediction significantly (51% vs 15%), demonstrating the value of multiple predictors in capturing complex relationships.

### Key Features

- **Clear mathematical notation** matching theory to code
- **Iterative convergence tracking** with cost history
- **Visualization** of cost function descent
- **Modular design** with reusable functions
- **Both single and multiple feature implementations** for comparison

### Usage

#### Single Feature

```python
# Load data
diabetes = load_diabetes()
X = diabetes.data[:, 2]  # BMI feature
y = diabetes.target

# Initialize parameters
w_init = 0
b_init = 0

# Train
w_final, b_final, J_history, p_history = gradient_descent(
    X, y, w_init, b_init, 
    alpha=0.01, 
    num_iters=3000, 
    compute_cost, 
    compute_gradient
)

# Predict
predictions = w_final * X + b_final
```

#### Multiple Features

```python
# Load data
diabetes = load_diabetes()
X = diabetes.data  # All 10 features
y = diabetes.target

# Initialize parameters
w_init = np.zeros(X.shape[1])
b_init = 0

# Train
w_final, b_final = gradient_descent(
    X, y, w_init, b_init,
    compute_cost,
    compute_gradient,
    alpha=0.01,
    num_iters=10000
)

# Predict
predictions = predict(X, w_final, b_final)
```

### Requirements

```bash
pip install numpy matplotlib scikit-learn
```

### Educational Purpose

These implementations prioritize **clarity and understanding** over production-ready performance. They demonstrate:
- How linear regression works mathematically
- The mechanics of gradient descent optimization
- Difference between single and multiple feature regression
- Impact of feature selection on model performance
- MSE minimization from first principles

---

## 🇧🇷 Português

Implementações do zero de Regressão Linear usando NumPy, demonstrando regressão com uma única feature e múltiplas features no dataset Diabetes.

### Visão Geral

Estes notebooks implementam algoritmos de regressão linear manualmente, focando na compreensão dos fundamentos matemáticos da otimização por gradiente descendente para problemas de regressão.

### Notebooks

1. **Linear_Regression_Single_Feature_From_Scratch.ipynb**
   - Regressão linear simples com uma feature (IMC)
   - Demonstração visual da otimização da função de custo
   - Gradiente descendente passo a passo

2. **Multiple_Linear_Regression_From_Scratch.ipynb**
   - Regressão linear múltipla com 10 features
   - Demonstra o tratamento de múltiplos preditores
   - Escalonamento de features e análise de convergência

### Dataset

**Dataset Diabetes** (de sklearn.datasets)
- **442 amostras** de pacientes com diabetes
- **10 features base**: idade, sexo, IMC, pressão arterial e 6 medições de soro sanguíneo
- **Alvo**: Medida quantitativa da progressão da doença um ano após o baseline

**Features usadas**:
- Feature Única: IMC (Índice de Massa Corporal) apenas
- Múltiplas Features: Todas as 10 variáveis base

### Detalhes da Implementação

#### ✅ Implementado do Zero

- **Função de custo**: Erro Quadrático Médio (MSE)
- **Cálculo do gradiente**: Cálculo de derivadas parciais
- **Gradiente descendente**: Algoritmo de otimização iterativa
- **Função de predição**: Avaliação do modelo linear
- **Métricas de performance**: Cálculo manual de R²

#### ⚠️ Dependências Externas (Não-Algorítmicas)

As seguintes utilidades do sklearn são usadas **apenas para carregamento de dados e avaliação**:
- `load_diabetes`: Carregamento do dataset
- `r2_score`: Métrica R² para comparação de modelos

**Nota**: O algoritmo central de regressão (custo, gradiente, otimização) é 100% do zero.

### Fundamentos Matemáticos

#### Modelo Linear
```
f(x) = w·x + b
```
- Feature única: `f(x) = w*x + b`
- Múltiplas features: `f(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`

#### Função de Custo (MSE)
```
J(w,b) = 1/(2m) * Σ(f(x) - y)²
```

#### Atualização do Gradiente Descendente
```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

Onde:
- `∂J/∂w = 1/m * Σ(f(x) - y) * x`
- `∂J/∂b = 1/m * Σ(f(x) - y)`

### Resultados

#### Regressão Linear com Feature Única

- **Custo Final (MSE)**: 2.531,50
- **RMSE**: 50,31 pontos
- **R²**: 0,1462 (14,62% de variância explicada)
- **Treinamento**: 3.000 iterações com α=0,01

**Análise**: Uma única feature (IMC) explica apenas ~15% da progressão da doença, mostrando a limitação de usar um único preditor.

#### Regressão Linear Múltipla

- **Custo Final (MSE)**: 1.452,19
- **R²**: 0,5102 (51,02% de variância explicada)
- **Treinamento**: 10.000 iterações com α=0,01
- **Features**: 10 variáveis base

**Análise**: Usar todas as 10 features melhora significativamente a predição (51% vs 15%), demonstrando o valor de múltiplos preditores para capturar relações complexas.

### Características Principais

- **Notação matemática clara** conectando teoria ao código
- **Rastreamento de convergência iterativa** com histórico de custo
- **Visualização** do descenso da função de custo
- **Design modular** com funções reutilizáveis
- **Implementações com uma e múltiplas features** para comparação

### Uso

#### Feature Única

```python
# Carregar dados
diabetes = load_diabetes()
X = diabetes.data[:, 2]  # Feature IMC
y = diabetes.target

# Inicializar parâmetros
w_init = 0
b_init = 0

# Treinar
w_final, b_final, J_history, p_history = gradient_descent(
    X, y, w_init, b_init, 
    alpha=0.01, 
    num_iters=3000, 
    compute_cost, 
    compute_gradient
)

# Predizer
predictions = w_final * X + b_final
```

#### Múltiplas Features

```python
# Carregar dados
diabetes = load_diabetes()
X = diabetes.data  # Todas as 10 features
y = diabetes.target

# Inicializar parâmetros
w_init = np.zeros(X.shape[1])
b_init = 0

# Treinar
w_final, b_final = gradient_descent(
    X, y, w_init, b_init,
    compute_cost,
    compute_gradient,
    alpha=0.01,
    num_iters=10000
)

# Predizer
predictions = predict(X, w_final, b_final)
```

### Requisitos

```bash
pip install numpy matplotlib scikit-learn
```

### Propósito Educacional

Estas implementações priorizam **clareza e compreensão** ao invés de performance pronta para produção. Elas demonstram:
- Como a regressão linear funciona matematicamente
- A mecânica da otimização por gradiente descendente
- Diferença entre regressão com uma e múltiplas features
- Impacto da seleção de features na performance do modelo
- Minimização do MSE desde os primeiros princípios

---

## Comparison | Comparação

| Metric / Métrica | Single Feature / Feature Única | Multiple Features / Múltiplas Features |
|------------------|--------------------------------|----------------------------------------|
| **Features** | 1 (BMI / IMC) | 10 (all / todas) |
| **MSE** | 2,531.50 | 1,452.19 |
| **R²** | 0.1462 (14.62%) | 0.5102 (51.02%) |
| **Iterations / Iterações** | 3,000 | 10,000 |
| **Improvement / Melhoria** | Baseline | **+249% variance explained** |

**Key Insight / Insight Principal**: Multiple features capture **3.5x more variance** than single feature, showing the importance of feature engineering.

Múltiplas features capturam **3,5x mais variância** que uma única feature, mostrando a importância da engenharia de features.

---

## License | Licença

Educational project - Free to use and modify.

Projeto educacional - Livre para usar e modificar.
