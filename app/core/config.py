"""
Configurações centralizadas do HealthIA Backend
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Configurações da aplicação"""
    
    # API Settings
    APP_NAME: str = "HealthIA API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "API para diagnóstico médico baseado em sintomas usando Machine Learning"
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "https://healthia.vercel.app",  # Atualizar com seu domínio real
    ]
    
    # Model Settings
    MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model")
    MODEL_FILE: str = "modelo_HealthIA.json"
    VECTORIZER_FILE: str = "vetorizador_HealthIA.pkl"
    ENCODER_FILE: str = "encoder_HealthIA.pkl"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        case_sensitive = True


settings = Settings()