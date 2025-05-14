from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate
from weaviate.connect import ConnectionParams
from weaviate import WeaviateClient
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def connect_to_weaviate():
    """
    Connect to Weaviate instance.
    """

    try:
        client = WeaviateClient(
            connection_params=ConnectionParams.from_url(
                url="http://weaviate:8080",  # Trong Docker
                grpc_port=50051  # Cổng gRPC mặc định của Weaviate
            )
        )
        client.connect()
        logger.info("🔸Kết nối tới Weaviate thành công.")
        return client
    except Exception as e:
        logger.error(f"🔸Lỗi kết nối tới Weaviate: {e}")
        client.close()
        return None
