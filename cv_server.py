import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from services.cv.processor import run_cv_task  

app = FastAPI()


TEMP_DIR = "/tmp/cv_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...), 
    mode: str = Form(...), 
    points: str = Form(None)
):
    
    temp_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
    
        processed_path, detected_class, confidence = run_cv_task(temp_path, mode, points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return {
        "processed_media_path": processed_path,
        "detected_class": detected_class,
        "confidence": confidence
    }