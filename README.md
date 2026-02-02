# 🏥 HealthIA Backend

Backend da aplicação HealthIA - API REST para diagnóstico médico baseado em sintomas usando Machine Learning.

## 🎯 Sobre o Projeto

O HealthIA Backend é uma API construída com FastAPI que utiliza um modelo XGBoost treinado para diagnosticar doenças com base em sintomas fornecidos pelo usuário. A API processa texto em linguagem natural, vetoriza os sintomas usando TF-IDF e retorna um diagnóstico com grau de confiança.

### Tecnologias Utilizadas

- **FastAPI** - Framework web moderno e rápido
- **XGBoost** - Algoritmo de Machine Learning
- **Scikit-learn** - Vetorização TF-IDF e preprocessing
- **Pydantic** - Validação de dados
- **Uvicorn** - Servidor ASGI

## 📁 Estrutura do Projeto

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # Aplicação principal FastAPI
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # Rotas da API
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Schemas Pydantic
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ml_service.py    # Serviço de ML
│   │   └── dataset.py       # Dataset de treinamento
│   └── core/
│       ├── __init__.py
│       └── config.py        # Configurações
├── model/                   # Arquivos do modelo treinado
│   ├── modelo_HealthIA.json
│   ├── vetorizador_HealthIA.pkl
│   └── encoder_HealthIA.pkl
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. **Clone o repositório** (se ainda não clonou)
```bash
git clone https://github.com/seu-usuario/healthia.git
cd healthia/backend
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
```

3. **Ative o ambiente virtual**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

4. **Instale as dependências**
```bash
pip install -r requirements.txt
```

5. **Configure as variáveis de ambiente**
```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o .env conforme necessário
```

6. **Execute o servidor**
```bash
# Desenvolvimento (com auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Ou simplesmente
python -m uvicorn app.main:app --reload
```

O servidor estará rodando em `http://localhost:8000`

## 📚 Documentação da API

Após iniciar o servidor, acesse:

- **Swagger UI (interativa)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Principais

#### `GET /api/v1/`
Endpoint raiz - Retorna informações básicas da API

#### `GET /api/v1/health`
Health check - Verifica se a API está funcionando

**Resposta:**
```json
{
  "status": "healthy",
  "app_name": "HealthIA API",
  "version": "1.0.0",
  "model_loaded": true
}
```

#### `GET /api/v1/diseases`
Lista todas as doenças que o modelo pode diagnosticar

**Resposta:**
```json
{
  "total_diseases": 20,
  "diseases": [
    "Anemia Falciforme",
    "Artrite Reumatoide",
    "Diabetes Tipo 1",
    ...
  ]
}
```

#### `POST /api/v1/predict`
**Endpoint principal** - Diagnostica sintomas

**Request:**
```json
{
  "symptoms": "febre alta, dor no corpo, cansaço extremo"
}
```

**Response:**
```json
{
  "diagnosis": "Febre Maculosa",
  "confidence": 92.5,
  "symptoms_received": ["febre", "alta", "dor", "no", "corpo", "cansaço", "extremo"],
  "recommendations": "⚠️ IMPORTANTE: Este é um diagnóstico automático..."
}
```

## 🧪 Testando a API

### Usando cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Diagnóstico
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "febre alta, dor no corpo, cansaço"}'
```

### Usando Python requests

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"symptoms": "febre alta, dor no corpo, cansaço"}
)

print(response.json())
```

## 🔧 Configuração (config.py)

As configurações são centralizadas em `app/core/config.py`:

- `APP_NAME`: Nome da aplicação
- `APP_VERSION`: Versão
- `ALLOWED_ORIGINS`: Origens permitidas para CORS
- `MODEL_PATH`: Caminho para os arquivos do modelo
- `HOST` e `PORT`: Configurações do servidor

## 🤖 Como Funciona o Modelo

1. **Recebimento**: API recebe sintomas em texto
2. **Preprocessing**: Limpa e normaliza o texto
3. **Vetorização**: Converte texto em números usando TF-IDF
4. **Predição**: Modelo XGBoost analisa e retorna diagnóstico
5. **Decodificação**: Converte número em nome da doença
6. **Resposta**: Retorna diagnóstico com confiança

## 🧑‍💻 Desenvolvimento

### Estrutura de Arquivos Explicada

- **`main.py`**: Ponto de entrada, configura FastAPI e CORS
- **`routes.py`**: Define todos os endpoints da API
- **`schemas.py`**: Validação de dados com Pydantic
- **`ml_service.py`**: Lógica de carregamento e uso do modelo
- **`dataset.py`**: Dados de treinamento e informações
- **`config.py`**: Configurações centralizadas

### Adicionando Novas Doenças

1. Adicione exemplos em `app/services/dataset.py`
2. Re-treine o modelo
3. Substitua os arquivos em `model/`
4. Reinicie o servidor

## 📦 Deploy

### Railway / Render

1. Configure as variáveis de ambiente no dashboard
2. Conecte o repositório Git
3. O deploy será automático

### Variáveis de Ambiente em Produção

```
APP_NAME=HealthIA API
HOST=0.0.0.0
PORT=8000
ALLOWED_ORIGINS=https://seu-frontend.vercel.app
```

## ⚠️ Avisos Importantes

- Este sistema é para fins educacionais
- **NÃO substitui consulta médica real**
- Sempre inclua disclaimers nas respostas
- Não use em produção sem validação médica adequada

## 📄 Licença

Este projeto é de código aberto para fins educacionais.

---

Desenvolvido por Mariana Santos Carminate | 2025
