from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings

app = FastAPI(
    title="API de Previsão de Ações",
    description="API para prever preços de ações usando modelo LSTM",
    version="1.0.0"
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substituir por origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Importar e incluir rotas
from .api.endpoints import predict
app.include_router(predict.router, prefix="/api/v1", tags=["predict"])

@app.get("/")
async def root():
    return {
        "message": "Bem-vindo à API de Previsão de Ações",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
