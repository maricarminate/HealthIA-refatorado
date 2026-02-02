"""
MAIN.PY - Arquivo Principal do Backend

EXPLICAÇÃO COMPLETA:
Este é o "coração" da aplicação. É aqui que:
1. Criamos a aplicação FastAPI
2. Configuramos CORS (para o frontend poder acessar)
3. Registramos todas as rotas
4. Configuramos middleware
5. Iniciamos o servidor

ANALOGIA:
Se o backend fosse um restaurante:
- main.py = O gerente que abre o restaurante
- Configura as mesas (rotas)
- Define as regras (CORS, middleware)
- Abre as portas (inicia o servidor)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.config import settings
from app.api import router

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    """
    Factory function para criar a aplicação FastAPI.
    
    EXPLICAÇÃO:
    Esta função cria e configura a aplicação FastAPI.
    Usamos uma função ao invés de criar direto para facilitar testes.
    
    Returns:
        FastAPI: Aplicação configurada
    """
    
    # PASSO 1: Criar aplicação FastAPI
    # EXPLICAÇÃO:
    # FastAPI é o framework web que usamos.
    # Passamos metadata como título, descrição, versão...
    # Isso aparece automaticamente na documentação (/docs)
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url="/docs",  # Swagger UI - documentação interativa
        redoc_url="/redoc",  # ReDoc - documentação alternativa
    )
    
    # PASSO 2: Configurar CORS
    # EXPLICAÇÃO DETALHADA DE CORS:
    # CORS = Cross-Origin Resource Sharing
    # 
    # Por padrão, navegadores BLOQUEIAM requisições de:
    # Frontend (http://localhost:3000) → Backend (http://localhost:8000)
    # 
    # Por quê? Segurança! Evita sites maliciosos acessarem APIs.
    # 
    # Mas NOSSO frontend PRECISA acessar nosso backend!
    # Então configuramos CORS para PERMITIR isso.
    # 
    # Em produção:
    # - Frontend: https://healthia.vercel.app
    # - Backend: https://healthia-api.railway.app
    # Também precisamos liberar!
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,  # Quais origens podem acessar
        allow_credentials=True,  # Permite enviar cookies
        allow_methods=["*"],  # Permite todos os métodos HTTP (GET, POST, etc)
        allow_headers=["*"],  # Permite todos os headers
    )
    
    logger.info(f"CORS configurado para: {settings.ALLOWED_ORIGINS}")
    
    # PASSO 3: Registrar rotas
    # EXPLICAÇÃO:
    # Aqui "conectamos" todas as rotas que definimos em routes.py
    # O prefix="/api/v1" significa que todas as rotas começam com /api/v1
    # Exemplo: GET / vira GET /api/v1/
    #          POST /predict vira POST /api/v1/predict
    # 
    # Versionamento (v1, v2, v3...) é boa prática:
    # - Permite criar novas versões sem quebrar clientes antigos
    # - /api/v1/predict continua funcionando
    # - Novos recursos em /api/v2/predict
    
    app.include_router(
        router,
        prefix="/api/v1",
        tags=["HealthIA"]  # Tag para agrupar na documentação
    )
    
    logger.info("Rotas registradas com sucesso!")
    
    # PASSO 4: Event handlers (opcional mas útil)
    # EXPLICAÇÃO:
    # Executam código em momentos específicos:
    # - startup: quando o servidor inicia
    # - shutdown: quando o servidor é desligado
    
    @app.on_event("startup")
    async def startup_event():
        """
        Executado quando o servidor inicia.
        
        EXPLICAÇÃO:
        Aqui você pode:
        - Conectar ao banco de dados
        - Carregar cache
        - Inicializar serviços externos
        - Etc.
        
        No nosso caso, só logamos que iniciou.
        O modelo ML já foi carregado em ml_service (import automático).
        """
        logger.info("=" * 70)
        logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} está iniciando...")
        logger.info(f"📚 Documentação disponível em: http://{settings.HOST}:{settings.PORT}/docs")
        logger.info(f"🏥 API disponível em: http://{settings.HOST}:{settings.PORT}/api/v1")
        logger.info("=" * 70)
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """
        Executado quando o servidor é desligado.
        
        EXPLICAÇÃO:
        Aqui você pode:
        - Fechar conexões com banco de dados
        - Salvar cache
        - Limpar recursos
        - Etc.
        """
        logger.info("🛑 Servidor sendo desligado...")
    
    # PASSO 5: Exception handlers (tratamento de erros global)
    # EXPLICAÇÃO:
    # Se algum erro não tratado acontecer em QUALQUER rota,
    # este handler captura e retorna uma resposta JSON amigável
    # ao invés de deixar o servidor crashar.
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """
        Handler global para exceções não tratadas.
        
        EXPLICAÇÃO:
        Último "safety net" (rede de segurança).
        Se algo der errado e não foi tratado nas rotas,
        cai aqui e retorna erro 500 formatado.
        """
        logger.error(f"Erro não tratado: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Erro interno do servidor",
                "detail": "Um erro inesperado ocorreu. Por favor, tente novamente."
            }
        )
    
    return app


# CRIAR INSTÂNCIA DA APLICAÇÃO
# EXPLICAÇÃO:
# Esta variável 'app' é o que será executado pelo Uvicorn
# Uvicorn é o servidor ASGI que roda FastAPI
# Comando: uvicorn app.main:app --reload
#          ↑ arquivo ↑ variável
app = create_application()


# PONTO DE ENTRADA (quando executa diretamente)
# EXPLICAÇÃO:
# Se você rodar: python app/main.py
# Este bloco executa e inicia o servidor
# 
# Mas o normal é usar: uvicorn app.main:app --reload
# Aí este bloco NÃO executa
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Iniciando servidor via Python direto...")
    logger.info("(Recomendado usar: uvicorn app.main:app --reload)")
    
    # Iniciar servidor Uvicorn
    uvicorn.run(
        "app.main:app",  # Caminho para a aplicação
        host=settings.HOST,  # 0.0.0.0 = aceita conexões de qualquer IP
        port=settings.PORT,  # 8000
        reload=True,  # Auto-reload quando código mudar (desenvolvimento)
        log_level="info"
    )