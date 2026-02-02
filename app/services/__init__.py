"""
Service de Machine Learning - Carrega e executa o modelo HealthIA

EXPLICAÇÃO DETALHADA:
Este é o CÉREBRO do sistema. Aqui acontece:
1. Carregamento do modelo treinado (XGBoost)
2. Carregamento do vetorizador (TF-IDF)
3. Carregamento do encoder (converte números em nomes de doenças)
4. Predição de diagnóstico baseado em sintomas

ANALOGIA:
Imagine um médico experiente:
- Modelo = conhecimento médico dele
- Vetorizador = como ele entende os sintomas que você descreve
- Encoder = dicionário que traduz diagnóstico técnico para nome da doença
"""

import os
import joblib
import xgboost as xgb
import numpy as np
from typing import Dict, List, Tuple
import logging

from app.core.config import settings
from app.services.dataset import get_available_diseases

# Configurar logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModelService:
    """
    Serviço que gerencia o modelo de Machine Learning.
    
    EXPLICAÇÃO:
    Esta classe é um "gerenciador" do modelo ML.
    Ela carrega todos os arquivos necessários UMA VEZ quando o servidor inicia,
    e depois fica disponível para fazer predições rapidamente.
    
    É como ter um médico sempre disponível, sem precisar "treinar" ele toda vez.
    """
    
    def __init__(self):
        """
        Inicializa o serviço carregando o modelo e componentes.
        
        EXPLICAÇÃO:
        Quando criamos uma instância desta classe (geralmente ao iniciar o servidor),
        este método __init__ é chamado automaticamente e carrega:
        - O modelo XGBoost treinado
        - O vetorizador TF-IDF
        - O encoder de labels
        """
        logger.info("Inicializando serviço de ML...")
        
        # Caminhos dos arquivos do modelo
        self.model_path = os.path.join(settings.MODEL_PATH, settings.MODEL_FILE)
        self.vectorizer_path = os.path.join(settings.MODEL_PATH, settings.VECTORIZER_FILE)
        self.encoder_path = os.path.join(settings.MODEL_PATH, settings.ENCODER_FILE)
        
        # Carregar componentes
        self._load_model()
        self._load_vectorizer()
        self._load_encoder()
        
        logger.info("✓ Serviço de ML inicializado com sucesso!")
    
    def _load_model(self):
        """
        Carrega o modelo XGBoost treinado.
        
        EXPLICAÇÃO:
        XGBoost é o algoritmo de ML que usamos.
        Ele foi treinado com sintomas e aprendeu a reconhecer padrões.
        Aqui apenas carregamos o modelo já treinado (não treinamos de novo).
        """
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            logger.info(f"✓ Modelo carregado de: {self.model_path}")
        except Exception as e:
            logger.error(f"✗ Erro ao carregar modelo: {str(e)}")
            raise
    
    def _load_vectorizer(self):
        """
        Carrega o vetorizador TF-IDF.
        
        EXPLICAÇÃO:
        TF-IDF converte texto em números.
        Por quê? Modelos ML só entendem números, não palavras.
        
        Exemplo:
        "febre alta" → [0.0, 0.8, 0.0, 0.6, ...]  (vetor de números)
        
        O vetorizador foi "treinado" junto com o modelo para saber
        como converter sintomas em números da mesma forma.
        """
        try:
            self.vectorizer = joblib.load(self.vectorizer_path)
            logger.info(f"✓ Vetorizador carregado de: {self.vectorizer_path}")
        except Exception as e:
            logger.error(f"✗ Erro ao carregar vetorizador: {str(e)}")
            raise
    
    def _load_encoder(self):
        """
        Carrega o encoder de labels.
        
        EXPLICAÇÃO:
        O modelo retorna números (0, 1, 2, 3...).
        O encoder traduz esses números para nomes de doenças.
        
        Exemplo:
        Modelo retorna: 5
        Encoder traduz: 5 → "Diabetes Tipo 1"
        """
        try:
            self.encoder = joblib.load(self.encoder_path)
            logger.info(f"✓ Encoder carregado de: {self.encoder_path}")
        except Exception as e:
            logger.error(f"✗ Erro ao carregar encoder: {str(e)}")
            raise
    
    def predict(self, symptoms: str) -> Dict:
        """
        Faz a predição de diagnóstico baseado em sintomas.
        
        EXPLICAÇÃO DO FLUXO:
        1. Recebe sintomas como texto: "febre alta, dor no corpo"
        2. Vetoriza (converte em números)
        3. Passa pelo modelo ML
        4. Modelo retorna um número
        5. Encoder traduz número para nome da doença
        6. Retorna resultado formatado
        
        Args:
            symptoms (str): String com sintomas (ex: "febre alta dor no corpo")
        
        Returns:
            Dict com:
                - diagnosis: nome da doença
                - confidence: confiança da predição (0-100%)
                - symptoms_processed: sintomas que foram processados
        """
        try:
            logger.info(f"Processando sintomas: {symptoms}")
            
            # PASSO 1: Limpar e preparar sintomas
            symptoms_cleaned = self._preprocess_symptoms(symptoms)
            
            # PASSO 2: Vetorizar sintomas (texto → números)
            symptoms_vectorized = self.vectorizer.transform([symptoms_cleaned])
            logger.info(f"Sintomas vetorizados: shape = {symptoms_vectorized.shape}")
            
            # PASSO 3: Fazer predição com o modelo
            # predict() retorna o número da classe
            prediction = self.model.predict(symptoms_vectorized)
            
            # PASSO 4: Obter probabilidades (confiança)
            # predict_proba() retorna array com probabilidade de cada classe
            prediction_proba = self.model.predict_proba(symptoms_vectorized)
            confidence = float(np.max(prediction_proba) * 100)  # Converter para porcentagem
            
            # PASSO 5: Decodificar número → nome da doença
            diagnosis = self.encoder.classes_[int(prediction[0])]
            
            logger.info(f"✓ Diagnóstico: {diagnosis} (confiança: {confidence:.2f}%)")
            
            # PASSO 6: Retornar resultado formatado
            return {
                "diagnosis": diagnosis,
                "confidence": round(confidence, 2),
                "symptoms_processed": symptoms_cleaned.split(),
                "all_probabilities": self._get_top_predictions(prediction_proba[0], top_n=3)
            }
            
        except Exception as e:
            logger.error(f"✗ Erro na predição: {str(e)}")
            raise Exception(f"Erro ao processar diagnóstico: {str(e)}")
    
    def _preprocess_symptoms(self, symptoms: str) -> str:
        """
        Limpa e prepara os sintomas para processamento.
        
        EXPLICAÇÃO:
        Remove espaços extras, converte para minúsculas, etc.
        Garante que os sintomas estejam no mesmo formato usado no treinamento.
        
        Args:
            symptoms: String bruta com sintomas
        
        Returns:
            String limpa e padronizada
        """
        # Converter para minúsculas
        symptoms_lower = symptoms.lower()
        
        # Remover vírgulas e substituir por espaços
        symptoms_clean = symptoms_lower.replace(",", " ")
        
        # Remover espaços extras
        symptoms_clean = " ".join(symptoms_clean.split())
        
        return symptoms_clean
    
    def _get_top_predictions(self, probabilities: np.ndarray, top_n: int = 3) -> List[Dict]:
        """
        Retorna as top N predições com suas probabilidades.
        
        EXPLICAÇÃO:
        Às vezes queremos ver não só o diagnóstico mais provável,
        mas também as outras possibilidades.
        
        Exemplo:
        1. Diabetes Tipo 1 (95%)
        2. Diabetes Tipo 2 (3%)
        3. Hipertireoidismo (1%)
        
        Args:
            probabilities: Array com probabilidades de todas as classes
            top_n: Quantas top predições retornar
        
        Returns:
            Lista com top N predições e suas probabilidades
        """
        # Pegar índices das top N probabilidades (ordenadas)
        top_indices = np.argsort(probabilities)[::-1][:top_n]
        
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                "disease": self.encoder.classes_[idx],
                "probability": round(float(probabilities[idx] * 100), 2)
            })
        
        return top_predictions
    
    def get_model_info(self) -> Dict:
        """
        Retorna informações sobre o modelo carregado.
        
        EXPLICAÇÃO:
        Útil para debug e health checks.
        Mostra se o modelo está carregado e pronto para uso.
        
        Returns:
            Dict com informações do modelo
        """
        return {
            "model_loaded": self.model is not None,
            "vectorizer_loaded": self.vectorizer is not None,
            "encoder_loaded": self.encoder is not None,
            "available_diseases": get_available_diseases(),
            "total_diseases": len(get_available_diseases())
        }


# INSTÂNCIA GLOBAL DO SERVIÇO
# EXPLICAÇÃO:
# Criamos UMA ÚNICA instância do serviço quando o módulo é importado.
# Isso evita carregar o modelo várias vezes (seria muito lento).
# Esta instância é compartilhada por todas as requisições.
ml_service = MLModelService()