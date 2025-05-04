from fastapi import APIRouter, Depends, HTTPException
from schemas.chat import QueryRequest, AnswerResponse, SourceDocument
from dependencies import get_current_user
import time
import uuid
from utils.utils import get_redis_history, save_chat_to_redis, search_term_in_dictionary,preprocess_vietnamese_query

from custom_output_parser import CustomOutputParser
import logging
import rag_components


logger = logging.getLogger(__name__)
def ask_question_service(app_state,chat_id: str, request: QueryRequest,user_email: str=Depends(get_current_user)):
    if not app_state['redis'].exists(f"chat:{chat_id}:meta"):
        raise HTTPException(status_code=404, detail="Chat ID not found")
    user_in_redis = app_state["redis"].hget(f"chat:{chat_id}:meta", "user")
    if not user_in_redis:
        raise HTTPException(status_code=404, detail="Chat not found")

    if user_in_redis.decode() != user_email:
        raise HTTPException(status_code=403, detail="Unauthorized")


    start_time = time.time()
    question = preprocess_vietnamese_query(request.question)['accented']


    if not app_state.get("qa_chain"):
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain chưa sẵn sàng.")


    term_result = search_term_in_dictionary(question, app_state['dict'])

    try:

        if term_result:
            answer_def = term_result.get("definition", "Không thể tạo câu trả lời.")
            save_chat_to_redis(app_state['redis'], chat_id, question, answer_def)
            return AnswerResponse(
                answer=answer_def,
                sources='legal_terms',
                processing_time=0.0
            )



        # Get chat history
        chat_history = get_redis_history(app_state['redis'], chat_id)

        # New code date: 2025-04-30
        # expanded_query = expand_query(question, llm=app_state["llm"])
        # retrieved_docs = app_state["retriever"].get_relevant_documents(expanded_query)
        # final_docs = rerank(expanded_query, retrieved_docs,top_k=5, reranker=app_state["reranker"])
        # Create a new QA chain for this specific chat session
        qa_chain = rag_components.create_qa_chain(
            app_state["llm"],
            app_state["vectorstore"],
            app_state["retriever"],
            chat_id=chat_id  # Pass the actual chat_id
        )


        if qa_chain is None:
            raise ValueError("Failed to create QA chain for this chat session")


        # Use the newly created chain
        raw_result = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })

        # Áp dụng CustomOutputParser để xử lý phần answer nếu cần
        parser = CustomOutputParser()
        parsed_result = parser.parse(raw_result["answer"])

        end_time = time.time()
        if isinstance(raw_result.get("answer"), dict):
            answer_data = raw_result["answer"]
            answer = answer_data.get("answer", "Không thể tạo câu trả lời.")
            source_documents = answer_data.get("source_documents", None)
        else:
            answer = raw_result.get("answer", "Không thể tạo câu trả lời.")
            source_documents = None


        print("Source documents:", raw_result.get("source_documents"))
        # Lưu vào Redis
        save_chat_to_redis(app_state['redis'],chat_id, question, answer)
        sources_list = []
        if raw_result.get("source_documents"):
            for doc in raw_result["source_documents"]:
                 sources_list.append(SourceDocument(
                     source=doc.metadata.get('source', 'N/A'),
                     page_content_preview=doc.page_content[:200] + "..."
                 ))
        return AnswerResponse(
            answer=parsed_result['highlight'],
            sources=sources_list if sources_list else None,
            processing_time=round(end_time - start_time, 2)
        )
    except Exception as e:
        print(f"Error during QA Chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")