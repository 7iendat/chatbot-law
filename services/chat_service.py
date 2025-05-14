from fastapi import Depends, HTTPException
from schemas.chat import QueryRequest, AnswerResponse, SourceDocument
from dependencies import get_current_user
import time
import json
from utils.utils import  save_chat_to_redis, search_term_in_dictionary,preprocess_vietnamese_query
# from langchain.memory import ConversationBufferMemory
import os
import logging
from langchain_community.chat_message_histories import RedisChatMessageHistory

logger = logging.getLogger(__name__)
def ask_question_service(app_state, request: QueryRequest,user_email: str=Depends(get_current_user)):
    chat_id = request.chat_id
    question = request.question
    if not app_state['redis'].exists(f"chat:{chat_id}:meta"):
        raise HTTPException(status_code=404, detail="Chat ID not found")
    user_in_redis = app_state["redis"].hget(f"chat:{chat_id}:meta", "user")
    if not user_in_redis:
        raise HTTPException(status_code=404, detail="Chat not found")

    if user_in_redis.decode() != user_email:
        raise HTTPException(status_code=403, detail="Unauthorized")


    start_time = time.time()
    question = preprocess_vietnamese_query(question)['accented']


    if not app_state.get("qa_chain"):
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain chưa sẵn sàng.")


    term_result = search_term_in_dictionary(question, app_state['dict'])

    try:
        # Initialize Redis memory for this chat_id
        redis_url = os.environ.get("REDIS_URL")
        if not redis_url:
            logger.error("REDIS_URL not set.")
            raise ValueError("Redis URL is required.")

        # memory = ConversationBufferMemory(
        #     memory_key="chat_history",
        #     chat_memory=RedisChatMessageHistory(url=redis_url, session_id=chat_id),
        #     return_messages=True,
        #     output_key="answer"
        # )

        chat_history = RedisChatMessageHistory(
            url=redis_url,
            session_id=chat_id
        )
        def get_chat_history(_):
            return chat_history.messages


        if term_result:
            answer_def = term_result.get("definition", "Không thể tạo câu trả lời.")
            save_chat_to_redis(app_state['redis'], chat_id, question, answer_def)
            return AnswerResponse(
                answer=answer_def,
                sources='legal_terms',
                processing_time=0.0
            )

        # Prepare input for qa_chain
        # input_data = {
        #     "chat_history": memory.load_memory_variables({}).get("chat_history", []),
        #     "question": question
        # }

        input_data = {
            "chat_history": chat_history.messages,
            "question": question
        }

        # Call qa_chain
        try:
            result = app_state["qa_chain"].invoke(input_data)
            logger.info(f"🔸QA Chain raw result for chat_id {chat_id}: {result}")
            # Handle unexpected output format
            if isinstance(result, dict) and "answer" in result:
                if isinstance(result["answer"], str):
                    try:
                        # Try parsing answer as JSON
                        parsed_answer = json.loads(result["answer"])
                        if isinstance(parsed_answer, dict) and "answer" in parsed_answer:
                            logger.warning(f"🔸Parsed JSON answer for chat_id {chat_id}: {parsed_answer}")
                            result = {"answer": parsed_answer["answer"]}
                        else:
                            logger.warning(f"🔸Answer is a JSON string but not a valid answer dict: {result['answer']}")
                    except json.JSONDecodeError:
                        # Not a JSON string, use as is
                        pass
                elif "question" in result:
                    logger.warning(f"🔸Unexpected 'question' key in result for chat_id {chat_id}: {result}")
                    result = {"answer": "Không thể xử lý câu hỏi. Vui lòng thử lại."}
                elif "raw" in result:
                    logger.warning(f"🔸Unexpected 'raw' key in result for chat_id {chat_id}: {result}")
                    result = {"answer": result["raw"]}


            if not isinstance(result, dict) or "answer" not in result:
                logger.error(f"🔸Invalid QA Chain result for chat_id {chat_id}: {result}")
                raise ValueError("QA Chain did not return a valid response.")

            response = result["answer"]
            if not isinstance(response, str):
                logger.warning(f"🔸Response is not a string for chat_id {chat_id}: {response}")
                response = str(response)
        except Exception as chain_error:
            logger.error(f"🔸QA Chain error for chat_id {chat_id}: {chain_error}")
            raise HTTPException(status_code=500, detail=f"QA Chain processing failed: {str(chain_error)}")

        # Save response to memory (for legal chain)
        # memory.save_context({"question": question}, {"answer": response})
        chat_history.add_user_message(question)
        chat_history.add_ai_message(response)
        end_time = time.time()

        # Lưu vào Redis
        save_chat_to_redis(app_state['redis'],chat_id, question, response)
        sources_list = []
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                 sources_list.append(SourceDocument(
                     source=doc.metadata.get('source', 'N/A'),
                     page_content_preview=doc.page_content[:200] + "..."
                 ))

        logger.info(f"🔸Question: {question}")
        logger.info(f"🔸Sources list:{sources_list}")
        logger.info("🔸Answer: {response}")

        return AnswerResponse(
            answer=response,
            sources=sources_list if sources_list else None,
            processing_time=round(end_time - start_time, 2)
        )
    except Exception as e:
        print(f"Error during QA Chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")