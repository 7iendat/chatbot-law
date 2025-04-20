from fastapi import APIRouter, Depends, HTTPException
from schemas.chat import QueryRequest, AnswerResponse, SourceDocument
from dependencies import get_current_user
import time
import uuid
from utils.utils import get_redis_history, save_chat_to_redis

def ask_question_service(app_state,request: QueryRequest,user=Depends(get_current_user)):
    start_time = time.time()
    question = request.question
    # Tự sinh session_id nếu người dùng không truyền hoặc truyền là "string"
    session_id = user['session_id'] or str(uuid.uuid4())
    if not app_state.get("qa_chain"):
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain chưa sẵn sàng.")

    print(f"Received question: {question}")
    try:
        raw_result = app_state["qa_chain"].invoke({
            "question": question,
            "chat_history": get_redis_history(app_state['redis'], session_id)
        })

        end_time = time.time()
        if isinstance(raw_result.get("answer"), dict):
            answer_data = raw_result["answer"]
            answer = answer_data.get("answer", "Không thể tạo câu trả lời.")
            source_documents = answer_data.get("source_documents", None)
        else:
            answer = raw_result.get("answer", "Không thể tạo câu trả lời.")
            source_documents = None


        # Lưu vào Redis
        save_chat_to_redis(app_state['redis'], session_id, question, answer)
        sources_list = []
        if raw_result.get("source_documents"):
            for doc in raw_result["source_documents"]:
                 sources_list.append(SourceDocument(
                     source=doc.metadata.get('source', 'N/A'),
                     page_content_preview=doc.page_content[:200] + "..."
                 ))
        return AnswerResponse(
            session_id=session_id,
            answer=answer,
            sources=sources_list if sources_list else None,
            processing_time=round(end_time - start_time, 2)
        )
    except Exception as e:
        print(f"Error during QA Chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")