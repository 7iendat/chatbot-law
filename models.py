from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field

from langchain_core.runnables import Runnable

from langchain_core.runnables import Runnable
from typing import Dict, Any

class WrappedLLMChain(Runnable):
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, input: Dict[str, Any], config: dict = None, **kwargs) -> Dict[str, Any]:
        response = self.chain.invoke(input, config=config, **kwargs)
        return {"answer": response}





# === Data models ===
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None, example=None)

class SourceDocument(BaseModel):
    source: str
    page_content_preview: str

class AnswerResponse(BaseModel):
    session_id: str
    answer: str
    sources: Optional[List[SourceDocument]] = None
    processing_time: float

class ChatHistoryItem(BaseModel):
    question: str
    answer: str

class ChatHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatHistoryItem]