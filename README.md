# Machine Learning Algorithms From Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-orange)
![Status](https://img.shields.io/badge/Status-Educational-green)

## English Description

### Overview
This repository contains fundamental Machine Learning algorithms (**Linear Regression** and **Logistic Regression**) implemented entirely from scratch using **Python** and **NumPy**. 

The goal of this project is not just to model data, but to open the "black box" of ML libraries. Instead of using high-level functions like `sklearn.fit()`, I implemented the underlying mathematics—including **Gradient Descent**, **Cost Functions**, and **Derivatives**—to understand exactly how models learn.

### Key Features & Technical Highlights
* **Vectorization:** Replaced inefficient Python `for` loops with NumPy matrix operations, improving training speed by over 100x.
* **L2 Regularization:** Implemented manual regularization (Ridge) to prevent overfitting on complex polynomial features.
* **Polynomial Regression:** Manual feature engineering to model non-linear relationships in data.
* **Weighted Loss Function:** Custom implementation of a weighted cost function to handle **imbalanced datasets** (e.g., Framingham Heart Study), significantly improving recall for minority classes.
* **Gradient Descent:** Step-by-step implementation of the optimization algorithm.

### Projects Included
1.  **Linear Regression (Diabetes Dataset):** Predicting disease progression based on BMI using a single-feature linear model. Focuses on visualizing the cost function convergence.
2.  **Logistic Regression (Heart Disease Prediction):** A complex implementation using the **Framingham dataset**. This project tackles data imbalance issues by implementing a custom weighted loss function and decision thresholds to maximize recall (sensitivity) for medical diagnosis.

### Technologies
* **Python** (Core Logic)
* **NumPy** (Linear Algebra & Vectorization)
* **Pandas** (Data Manipulation)
* **Matplotlib/Seaborn** (Visualization)

---

## Descrição em Português

### Visão Geral
Este repositório contém algoritmos fundamentais de Machine Learning (**Regressão Linear** e **Regressão Logística**) implementados totalmente do zero (From Scratch) utilizando **Python** e **NumPy**.

O objetivo deste projeto não é apenas modelar dados, mas abrir a "caixa preta" das bibliotecas de ML. Em vez de usar funções de alto nível como `sklearn.fit()`, implementei a matemática subjacente — incluindo **Gradiente Descendente**, **Funções de Custo** e **Derivadas** — para entender exatamente como os modelos aprendem.

### Destaques Técnicos
* **Vetorização:** Substituição de loops `for` ineficientes por operações matriciais com NumPy, acelerando o treinamento em mais de 100x.
* **Regularização L2:** Implementação manual de regularização para evitar *overfitting* em features polinomiais complexas.
* **Regressão Polinomial:** Engenharia de features manual para modelar relações não-lineares nos dados.
* **Função de Perda Ponderada (Weighted Loss):** Implementação customizada de uma função de custo ponderada para lidar com **datasets desbalanceados** (estudo de caso Framingham), melhorando significativamente o *recall* para classes minoritárias.
* **Gradiente Descendente:** Implementação passo a passo do algoritmo de otimização.

### Projetos Incluídos
1.  **Regressão Linear (Dataset Diabetes):** Previsão da progressão da diabetes baseada no IMC usando um modelo linear simples. Foco na visualização da convergência da função de custo.
2.  **Regressão Logística (Previsão de Doença Cardíaca):** Uma implementação complexa usando o dataset **Framingham**. Este projeto aborda problemas de desbalanceamento de dados implementando uma função de perda ponderada e ajuste de limiares de decisão para maximizar o *recall* (sensibilidade) para diagnósticos médicos.

### Tecnologias
* **Python** (Lógica Central)
* **NumPy** (Álgebra Linear & Vetorização)
* **Pandas** (Manipulação de Dados)
* **Matplotlib/Seaborn** (Visualização)

---

## How to Run / Como Rodar

1.  Clone the repository:
    ```bash
    git clone https://github.com/pedroandrade-tech/ml-algorithms-from-scratch.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd Machine-Learning-Algorithms-From-Scratch
    ```
3.  Install requirements:
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```
4.  Run the notebooks using Jupyter Lab or VS Code.

## Author
* **Pedro Andrade** - *Aspiring Machine Learning Engineer & Developer*
* https://www.linkedin.com/in/pedro-andrade-959214243/
