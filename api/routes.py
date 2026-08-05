from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session


from db.database import get_db
from db.models import DetectionResult

router = APIRouter()

@router.post("/detect")
async def detect_fish(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db) 
):
    image_bytes = await file.read()
    file_size_kb = len(image_bytes) / 1024

  
    new_detection = DetectionResult(
        image_path=file.filename,  
        fish_class="Test_ClownFish",
        confidence=0.99             
    )
    
    db.add(new_detection)
    db.commit()          
    db.refresh(new_detection)

    return {
        "status": "success",
        "detection_id": new_detection.id,
        "fish_class": new_detection.fish_class,
        "message": "Данные успешно сохранены в БД!"
    }


@router.get("/history")
def get_history(db: Session = Depends(get_db)):
    history = db.query(DetectionResult).all()
    return {"history": history}