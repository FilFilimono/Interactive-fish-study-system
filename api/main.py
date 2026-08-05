from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.routes import router as api_router

app = FastAPI(
    title="Interactive Fish Study API",
    description="Бэкенд для системы распознавания рыб и общения с ИИ",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api", tags=["Fish Detection & NLP"])

@app.get("/")
def read_root():
    
    return JSONResponse(
        content={"message": "Сервер Interactive Fish Study System успешно запущен!"},
        media_type="application/json; charset=utf-8"
    )

@app.get("/health")
def health_check():
    return JSONResponse(
        content={"status": "ok", "services": ["cv", "nlp", "db"]},
        media_type="application/json; charset=utf-8"
    )