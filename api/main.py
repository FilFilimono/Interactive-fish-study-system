import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr

from api.routes.router import router as api_router
from ui.app import app as gradio_ui

app = FastAPI(
    title="Interactive Fish Study API",
    description="Бэкенд для системы распознавания рыб и общения с ИИ",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DET_DIR = os.path.join(BASE_DIR, "data", "output", "detection")
os.makedirs(OUTPUT_DET_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=OUTPUT_DET_DIR), name="static")

app.include_router(api_router, prefix="/api", tags=["Fish Detection & NLP"])

@app.get("/health")
def health_check():
    return JSONResponse(
        content={"status": "ok", "services": ["cv", "nlp", "db", "ui"]},
        media_type="application/json; charset=utf-8"
    )
    
app = gr.mount_gradio_app(app, gradio_ui, path="/")