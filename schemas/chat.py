from pydantic import BaseModel,EmailStr
from typing import List, Optional
from pydantic import Field

class QueryRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    source: str
    page_content_preview: str

class AnswerResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceDocument]] = None
    processing_time: float


class ChatHistoryItem(BaseModel):
    question: str
    answer: str

class ChatHistoryResponse(BaseModel):
    chat_id: str
    history: List[ChatHistoryItem]