"""
Pydantic schemas para validação de requests e responses
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class SymptomsRequest(BaseModel):
    """Schema para requisição de diagnóstico"""
    symptoms: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Sintomas separados por vírgula ou espaços",
        examples=["febre alta, dor no corpo, cansaço extremo"]
    )
    
    @field_validator('symptoms')
    @classmethod
    def validate_symptoms(cls, v: str) -> str:
        """Validar e limpar sintomas"""
        if not v or v.strip() == "":
            raise ValueError("Sintomas não podem estar vazios")
        
        # Remover espaços extras
        v = " ".join(v.split())
        
        return v.lower()


class DiagnosisResponse(BaseModel):
    """Schema para resposta de diagnóstico"""
    diagnosis: str = Field(..., description="Doença diagnosticada")
    confidence: Optional[float] = Field(None, description="Confiança da predição (0-100%)")
    symptoms_received: List[str] = Field(..., description="Sintomas processados")
    recommendations: Optional[str] = Field(None, description="Recomendações gerais")
    
    class Config:
        json_schema_extra = {
            "example": {
                "diagnosis": "Diabetes Tipo 1",
                "confidence": 95.5,
                "symptoms_received": ["sede constante", "micção frequente", "perda de peso rápida"],
                "recommendations": "Consulte um médico imediatamente para confirmação do diagnóstico"
            }
        }


class HealthCheckResponse(BaseModel):
    """Schema para health check"""
    status: str
    app_name: str
    version: str
    model_loaded: bool


class AvailableDiseasesResponse(BaseModel):
    """Schema para lista de doenças disponíveis"""
    total_diseases: int
    diseases: List[str]


class ErrorResponse(BaseModel):
    """Schema para respostas de erro"""
    error: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Erro na predição",
                "detail": "Sintomas não reconhecidos pelo modelo"
            }
        }