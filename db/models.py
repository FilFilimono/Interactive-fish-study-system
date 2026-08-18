from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from db.database import Base

class DetectionResult(Base):
    __tablename__ = "detection_results" 

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=True) 
    fish_class = Column(String, index=True)    
    confidence = Column(Float)                 
    created_at = Column(DateTime, default=datetime.utcnow) 

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True) 
    role = Column(String)                   
    content = Column(Text)                  
    created_at = Column(DateTime, default=datetime.utcnow)