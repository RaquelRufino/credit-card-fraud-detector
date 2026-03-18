# Credit Card Fraud Detector

Este projeto apresenta uma implementação de Machine Learning voltada para a detecção de fraudes em transações de cartão de crédito. O foco principal é demonstrar a estruturação de modelos preditivos para que possam ser integrados de forma eficiente a sistemas de backend, saindo do ambiente de experimentação de notebooks para um formato de serviço.

A ideia central é tratar o modelo como um componente de software, com separação clara entre as etapas de treinamento, avaliação e exposição via API.

## Estrutura do Projeto

O código está organizado para separar as responsabilidades de desenvolvimento e produção:

- **src/training/**: Scripts para pré-processamento de dados, treinamento do modelo e geração de artefatos.
- **src/inference/**: Camada lógica responsável por carregar o modelo treinado e realizar as predições de forma isolada.
- **src/api/**: Implementação de interface HTTP utilizando FastAPI para permitir o consumo do modelo por outras aplicações.
- **notebooks/**: Registros da análise exploratória (EDA) e testes iniciais de algoritmos.

## Dataset

O projeto utiliza o dataset de transações reais disponível no Kaggle:
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Para executar o treinamento:

1. Faça o download do arquivo `creditcard.csv`.
2. Salve o arquivo no diretório `data/` na raiz do projeto.

## Instalação e Execução

### 1. Configuração do Ambiente

```bash
git clone [https://github.com/seu-usuario/credit-card-fraud-detector.git](https://github.com/seu-usuario/credit-card-fraud-detector.git)
cd credit-card-fraud-detector

# Criação e ativação do ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalação de dependências
pip install -r requirements.txt

```

### 2. Treinamento e Avaliação

Para processar os dados e salvar o modelo treinado:

```bash
python src/training/train_model.py
```

Para validar a performance do modelo gerado:

```bash
python src/training/evaluate_model.py
```

### 3. Execução da API

Para iniciar o servidor local e testar os endpoints de predição:

```bash
uvicorn src.api.main:app --reload
```

A documentação interativa (Swagger UI) pode ser acessada em: http://localhost:8000/docs

## Métricas e Performance

Dado que o dataset é altamente desbalanceado (transações fraudulentas são uma pequena fração do total), o projeto não utiliza a acurácia como métrica principal. O foco da avaliação está em:

Recall: Para garantir que o maior número possível de fraudes seja detectado.

Precision: Para evitar que transações legítimas sejam bloqueadas indevidamente.

F1-Score: Para encontrar o equilíbrio ideal entre as duas métricas anteriores.
