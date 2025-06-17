from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate
import weaviate.classes as wvc # Import các lớp của weaviate
from weaviate.connect import ConnectionParams
from weaviate import WeaviateClient
import logging
import config

logger = logging.getLogger(__name__)


def check_collection_stats(client):
    collection_name = config.WEAVIATE_COLLECTION_NAME

    # Kiểm tra xem collection có tồn tại không
    if not client.collections.exists(collection_name):
        logger.info(f"Lỗi: Collection '{collection_name}' không tồn tại trong Weaviate.")
        client.close()
        return

    # Lấy đối tượng collection
    collection = client.collections.get(collection_name)

    logger.info("\n--- Thống kê dữ liệu trong VectorDB ---")

    # 1. Đếm tổng số object
    try:
        total_count_response = collection.aggregate.over_all(total_count= True)
        logger.info(f"Tổng số chunk trong collection '{collection_name}': {total_count_response.total_count}")
    except Exception as e:
        logger.info(f"Lỗi khi đếm tổng số object: {e}")

    # 2. Đếm số lượng chunk cho từng lĩnh vực quan trọng
    fields_to_check = ["giao_thong", "hinh_su", "lao_dong", "dat_dai", "y_te", "khac"]
    for field in fields_to_check:
        try:
            # Tạo bộ lọc (filter) theo cú pháp v4
            field_filter = wvc.query.Filter.by_property("field").equal(field)

            # Thực hiện truy vấn aggregate với bộ lọc
            response = collection.aggregate.over_all(filters=field_filter, total_count=True)

            logger.info(f"Số lượng chunk có lĩnh vực '{field}': {response.total_count}")
        except Exception as e:
            # Lỗi có thể xảy ra nếu trường "field" không được index hoặc không tồn tại
            logger.info(f"Lỗi khi kiểm tra lĩnh vực '{field}': {e}")

    # 3. Lấy một ví dụ về tài liệu "giao thông" để xem chất lượng
    logger.info("\n--- Lấy ví dụ chunk 'giao_thong' ---")
    try:
        gt_filter = wvc.query.Filter.by_property("field").equal("giao_thong")
        response = collection.query.fetch_objects(
            limit=1,
            filters=gt_filter,
            return_properties=["title", "field", "source", "text"] # Thay "content" bằng tên trường text của bạn
        )

        if response.objects:
            first_gt_object = response.objects[0]
            logger.info("Ví dụ chunk giao thông tìm thấy:")
            logger.info(f"  Tiêu đề: {first_gt_object.properties.get('title')}")
            logger.info(f"  File nguồn: {first_gt_object.properties.get('source')}")
            content_preview = first_gt_object.properties.get('text', '')
            logger.info(f"  Nội dung: {content_preview[:300]}...")
        else:
            logger.info("Không tìm thấy chunk nào thuộc lĩnh vực 'giao_thong'.")

    except Exception as e:
        logger.info(f"Lỗi khi lấy ví dụ chunk 'giao_thong': {e}")

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

        check_collection_stats(client=client)

        logger.info("🔸Kết nối tới Weaviate thành công.")
        return client
    except Exception as e:
        logger.error(f"🔸Lỗi kết nối tới Weaviate: {e}")
        client.close()
        return None
