# core/runnables.py
import logging
from typing import Dict, Any, List
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence

logger = logging.getLogger(__name__)

def router_as_runnable(
    routes: dict[str, RunnableSequence],
    get_key: RunnableLambda,
    default: RunnableSequence = None
) -> RunnableLambda:
    """
    Tạo một RunnableLambda để định tuyến input đến một runnable khác dựa trên một key.
    """
    def dispatch(input_data_dict):
        key = get_key.invoke(input_data_dict)
        logging.info(f"🔸[Router] Route key: {key}")
        selected_runnable = routes.get(key, default)
        if selected_runnable is None:
            raise ValueError(f"No route found for key '{key}' and no default route is set.")
        logging.info(f"🔸[Router] Selected runnable type: {type(selected_runnable)}")
        return selected_runnable

    return RunnableLambda(dispatch)


class WrappedLLMChain(Runnable):
    """
    Một lớp bao bọc (wrapper) cho một chain của LangChain để chuẩn hóa output.
    """
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, input: Dict[str, Any], config: dict = None, **kwargs) -> Dict[str, Any]:
        logger.info(f"WrappedLLMChain input: {input}")
        response = self.chain.invoke(input, config=config, **kwargs)
        logger.info(f"WrappedLLMChain raw response: {response}")

        if isinstance(response, dict) and "answer" in response:
            result = {"answer": response["answer"]}
            if "source_documents" in response:
                result["source_documents"] = response["source_documents"]
        else:
            result = {"answer": str(response)}

        logger.info(f"WrappedLLMChain processed result: {result}")
        return result

    def with_config(self, config):
        self.chain.with_config(config)
        return self