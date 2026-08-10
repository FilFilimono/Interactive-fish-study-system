# import os
# import shutil
# import json
# from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
# from sqlalchemy.orm import Session
# import uuid
# from db.database import get_db
# from db.models import DetectionResult, ChatMessage
# from services.nlp.generate_response import get_initial_fact, get_chat_response
# from services.cv.processor import run_cv_task

# router = APIRouter()

# UPLOAD_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "uploads"))
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @router.post("/process_media")
# async def process_media(file: UploadFile = File(...), mode: str = Form(...), points: str = Form(None),db: Session = Depends(get_db)):
#     file_extension = os.path.splitext(file.filename)[1]
#     unique_filename = f"{uuid.uuid4().hex}{file_extension}"
#     file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     try:
#         processed_media_path, detected_class, confidence = run_cv_task(file_path, mode, points)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"ошибка работы CV: {str(e)}")

#     filename = os.path.basename(processed_media_path)
#     processed_media_url = f"/static/{filename}"

#     try:
#         nlp_result = get_initial_fact(detected_class)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"ошибка генерации ответа LLM: {str(e)}")

#     new_detection = DetectionResult(
#         image_path=processed_media_url,
#         fish_class=detected_class,
#         confidence=confidence
#     )
#     db.add(new_detection)
#     db.commit()
#     db.refresh(new_detection)

#     session_id = f"session_{new_detection.id}"
#     for msg in nlp_result["messages_history"]:
#         db.add(ChatMessage(session_id=session_id, role=msg.get("role"), content=msg.get("content")))
#     db.commit()

#     return {
#         "status": "success",
#         "session_id": session_id,
#         "processed_media_url": processed_media_url,
#         "fish_class": detected_class,
#         "initial_llm_reply": nlp_result["reply"]
#     }

# @router.post("/chat")
# async def chat_with_llm(
#     session_id: str = Form(...), 
#     user_message: str = Form(...),
#     db: Session = Depends(get_db)
# ):
    
#     chat_history_db = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.id).all()
#     if not chat_history_db:
#         raise HTTPException(status_code=404, detail="Сессия не найдена")

#     messages_history = [{"role": msg.role, "content": msg.content} for msg in chat_history_db]

#     try:
#         nlp_result = get_chat_response(messages_history, user_message)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     db.add(ChatMessage(session_id=session_id, role="user", content=user_message))
#     db.add(ChatMessage(session_id=session_id, role="assistant", content=nlp_result["reply"]))
#     db.commit()

#     return {"status": "success", "reply": nlp_result["reply"]}

import os
import shutil
import uuid
import httpx
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import DetectionResult, ChatMessage
from services.nlp.generate_response import get_initial_fact, get_chat_response

router = APIRouter()

# см. docker-compose.yml: environment CV_URL=http://cv:8001 у сервиса nlp
CV_URL = os.getenv("CV_URL", "http://cv:8001")

UPLOAD_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/process_media")
async def process_media(file: UploadFile = File(...), mode: str = Form(...), points: str = Form(None), db: Session = Depends(get_db)):
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # CV теперь в отдельном контейнере — шлём файл в cv_server.py по HTTP,
    # вместо прямого вызова run_cv_task() (его тут просто нет физически).
    try:
        with open(file_path, "rb") as f:
            async with httpx.AsyncClient(timeout=600) as client:
                cv_resp = await client.post(
                    f"{CV_URL}/analyze",
                    files={"file": (file.filename, f, file.content_type)},
                    data={"mode": mode, "points": points or ""},
                )
        cv_resp.raise_for_status()
        cv_data = cv_resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"ошибка работы CV: {e.response.text}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"ошибка работы CV: {str(e)}")

    processed_media_path = cv_data["processed_media_path"]
    detected_class = cv_data["detected_class"]
    confidence = cv_data["confidence"]

    filename = os.path.basename(processed_media_path)
    processed_media_url = f"/static/{filename}"

    try:
        nlp_result = get_initial_fact(detected_class)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ошибка генерации ответа LLM: {str(e)}")

    new_detection = DetectionResult(
        image_path=processed_media_url,
        fish_class=detected_class,
        confidence=confidence
    )
    db.add(new_detection)
    db.commit()
    db.refresh(new_detection)

    session_id = f"session_{new_detection.id}"
    for msg in nlp_result["messages_history"]:
        db.add(ChatMessage(session_id=session_id, role=msg.get("role"), content=msg.get("content")))
    db.commit()

    return {
        "status": "success",
        "session_id": session_id,
        "processed_media_url": processed_media_url,
        "fish_class": detected_class,
        "initial_llm_reply": nlp_result["reply"]
    }

@router.post("/chat")
async def chat_with_llm(
    session_id: str = Form(...),
    user_message: str = Form(...),
    db: Session = Depends(get_db)
):
    chat_history_db = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.id).all()
    if not chat_history_db:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    messages_history = [{"role": msg.role, "content": msg.content} for msg in chat_history_db]

    try:
        nlp_result = get_chat_response(messages_history, user_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    db.add(ChatMessage(session_id=session_id, role="user", content=user_message))
    db.add(ChatMessage(session_id=session_id, role="assistant", content=nlp_result["reply"]))
    db.commit()

    return {"status": "success", "reply": nlp_result["reply"]}