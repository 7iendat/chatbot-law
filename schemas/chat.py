from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime

class AppState(BaseModel):
    embeddings: Optional[Any] = None
    vectorstore: Optional[Any] = None
    llm: Optional[Any] = None
    process_input_llm: Optional[Any] = None
    qa_chain: Optional[Any] = None
    device: str = "cpu"
    google_api_key: Optional[str] = None
    dict: Dict = {}
    redis: Optional[Any] = None
    retriever: Optional[Any] = None
    weaviateDB: Optional[Any] = None
    reranker: Optional[Any] = None

class QueryRequest(BaseModel):
    chat_id: str
    input: str

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime

class SourceDocument(BaseModel):
    source: str
    page_content_preview: str

class AnswerResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceDocument]] = None
    processing_time: float

class ChatHistoryResponse(BaseModel): # Model cho response của API này
    chat_id: str
    history: List[Message]
    created_at: datetime
    updated_at: datetime
    user_id: str


class MessageItem(BaseModel):
    role: str
    content: str
    timestamp: datetime

class ConversationResponse(BaseModel):
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageItem]
