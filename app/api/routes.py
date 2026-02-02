"""
Rotas da API - Endpoints HTTP

EXPLICAÇÃO COMPLETA:
Este arquivo define TODOS os endpoints (URLs) que nossa API oferece.

ANALOGIA:
Pensa nas rotas como "portas" de um prédio:
- Porta 1 (GET /): Entrada principal, dá boas-vindas
- Porta 2 (POST /predict): Laboratório, recebe sintomas e retorna diagnóstico
- Porta 3 (GET /health): Recepção, mostra se tudo está funcionando
- Porta 4 (GET /diseases): Biblioteca, lista todas as doenças

Cada rota:
1. Recebe uma requisição HTTP
2. Valida os dados (usando Pydantic schemas)
3. Chama o serviço ML
4. Retorna a resposta formatada
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
import logging

from app.models.schemas import (
    SymptomsRequest,
    DiagnosisResponse,
    HealthCheckResponse,
    AvailableDiseasesResponse,
    ErrorResponse
)
from app.services import ml_service, get_available_diseases
from app.core.config import settings

# Configurar logging
logger = logging.getLogger(__name__)

# Criar router
# EXPLICAÇÃO: APIRouter agrupa rotas relacionadas
router = APIRouter()


@router.get(
    "/",
    summary="Bem-vindo à API",
    description="Endpoint raiz que retorna informações básicas da API"
)
async def root():
    """
    ENDPOINT RAIZ - GET /
    
    EXPLICAÇÃO:
    Este é o endpoint mais simples. Apenas dá boas-vindas.
    Útil para testar se a API está no ar.
    
    EXEMPLO DE USO:
    GET http://localhost:8000/
    
    RETORNA:
    {
        "message": "Bem-vindo ao HealthIA API",
        "version": "1.0.0",
        "docs": "/docs"
    }
    """
    return {
        "message": f"Bem-vindo ao {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",  # Link para documentação automática do FastAPI
        "description": settings.APP_DESCRIPTION
    }


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Verifica se a API e o modelo ML estão funcionando corretamente"
)
async def health_check():
    """
    ENDPOINT HEALTH CHECK - GET /health
    
    EXPLICAÇÃO:
    Verifica se tudo está funcionando:
    - API está no ar?
    - Modelo ML está carregado?
    - Todos os componentes estão OK?
    
    É como fazer um "check-up" da API.
    Muito usado por serviços de monitoramento.
    
    EXEMPLO DE USO:
    GET http://localhost:8000/health
    
    RETORNA:
    {
        "status": "healthy",
        "app_name": "HealthIA API",
        "version": "1.0.0",
        "model_loaded": true
    }
    """
    try:
        # Pegar informações do modelo
        model_info = ml_service.get_model_info()
        
        return HealthCheckResponse(
            status="healthy" if model_info["model_loaded"] else "unhealthy",
            app_name=settings.APP_NAME,
            version=settings.APP_VERSION,
            model_loaded=model_info["model_loaded"]
        )
    except Exception as e:
        logger.error(f"Health check falhou: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            app_name=settings.APP_NAME,
            version=settings.APP_VERSION,
            model_loaded=False
        )


@router.get(
    "/diseases",
    response_model=AvailableDiseasesResponse,
    summary="Listar Doenças",
    description="Retorna lista de todas as doenças que o modelo pode diagnosticar"
)
async def list_diseases():
    """
    ENDPOINT LISTAR DOENÇAS - GET /diseases
    
    EXPLICAÇÃO:
    Retorna todas as doenças que nosso modelo conhece/pode diagnosticar.
    Útil para o frontend mostrar ao usuário quais doenças são suportadas.
    
    EXEMPLO DE USO:
    GET http://localhost:8000/diseases
    
    RETORNA:
    {
        "total_diseases": 20,
        "diseases": [
            "Anemia Falciforme",
            "Artrite Reumatoide",
            "Diabetes Tipo 1",
            ...
        ]
    }
    """
    try:
        diseases = get_available_diseases()
        
        return AvailableDiseasesResponse(
            total_diseases=len(diseases),
            diseases=diseases
        )
    except Exception as e:
        logger.error(f"Erro ao listar doenças: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao buscar lista de doenças"
        )


@router.post(
    "/predict",
    response_model=DiagnosisResponse,
    summary="Diagnosticar Sintomas",
    description="Recebe sintomas e retorna diagnóstico baseado em ML",
    responses={
        200: {
            "description": "Diagnóstico realizado com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "diagnosis": "Diabetes Tipo 1",
                        "confidence": 95.5,
                        "symptoms_received": ["sede constante", "micção frequente", "perda de peso"],
                        "recommendations": "Consulte um médico para confirmação"
                    }
                }
            }
        },
        400: {
            "description": "Sintomas inválidos",
            "model": ErrorResponse
        },
        500: {
            "description": "Erro interno no servidor",
            "model": ErrorResponse
        }
    }
)
async def predict_diagnosis(request: SymptomsRequest):
    """
    ENDPOINT PRINCIPAL - POST /predict
    
    EXPLICAÇÃO DETALHADA:
    Este é o endpoint MAIS IMPORTANTE da API!
    É aqui que a mágica acontece.
    
    FLUXO:
    1. Frontend envia sintomas: {"symptoms": "febre alta, dor no corpo"}
    2. FastAPI valida automaticamente usando SymptomsRequest (Pydantic)
    3. Se válido, chama ml_service.predict()
    4. ml_service processa sintomas e retorna diagnóstico
    5. FastAPI formata resposta usando DiagnosisResponse
    6. Retorna JSON pro frontend
    
    VALIDAÇÕES AUTOMÁTICAS:
    - Sintomas não podem estar vazios
    - Mínimo 3 caracteres
    - Máximo 500 caracteres
    - Pydantic cuida disso automaticamente!
    
    EXEMPLO DE USO:
    POST http://localhost:8000/predict
    Body: {
        "symptoms": "febre alta, dor no corpo, cansaço extremo"
    }
    
    RETORNA:
    {
        "diagnosis": "Febre Maculosa",
        "confidence": 92.5,
        "symptoms_received": ["febre", "alta", "dor", "no", "corpo", "cansaço", "extremo"],
        "recommendations": "Este é um diagnóstico automático..."
    }
    """
    try:
        logger.info(f"Nova requisição de diagnóstico: {request.symptoms}")
        
        # PASSO 1: Chamar o serviço ML para fazer a predição
        prediction_result = ml_service.predict(request.symptoms)
        
        # PASSO 2: Adicionar recomendações padrão
        recommendations = (
            "⚠️ IMPORTANTE: Este é um diagnóstico automático baseado em Machine Learning. "
            "NÃO substitui consulta médica. "
            "Procure um profissional de saúde para confirmação e tratamento adequado."
        )
        
        # PASSO 3: Montar e retornar resposta
        return DiagnosisResponse(
            diagnosis=prediction_result["diagnosis"],
            confidence=prediction_result["confidence"],
            symptoms_received=prediction_result["symptoms_processed"],
            recommendations=recommendations
        )
        
    except ValueError as e:
        # Erro de validação (sintomas inválidos, etc)
        logger.error(f"Erro de validação: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        # Erro genérico (problema no modelo, etc)
        logger.error(f"Erro ao processar diagnóstico: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao processar diagnóstico. Tente novamente."
        )


@router.get(
    "/model-info",
    summary="Informações do Modelo",
    description="Retorna informações detalhadas sobre o modelo ML carregado"
)
async def get_model_info():
    """
    ENDPOINT INFO DO MODELO - GET /model-info
    
    EXPLICAÇÃO:
    Retorna informações técnicas sobre o modelo:
    - Está carregado?
    - Quantas doenças conhece?
    - Quais componentes estão ativos?
    
    Útil para debug e monitoramento.
    
    EXEMPLO DE USO:
    GET http://localhost:8000/model-info
    
    RETORNA:
    {
        "model_loaded": true,
        "vectorizer_loaded": true,
        "encoder_loaded": true,
        "total_diseases": 20,
        "available_diseases": [...]
    }
    """
    try:
        model_info = ml_service.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Erro ao buscar informações do modelo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao buscar informações do modelo"
        )