from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# VectorDB parameters
location = ":memory:"
collection_name = "my_kb"
vector_size = 3
distance_metric = models.Distance.COSINE
group_id = "kb_id"
response_limit = 1

client = QdrantClient(url="http://localhost:6333")


def collection_exists(collection_name: str) -> bool:
    try:
        client.get_collection(collection_name)
        return True
    except UnexpectedResponse as e:
        if e.status_code == 404:
            return False
        raise


def query_vector_in_group(gid, query_vector) -> models.QueryResponse:
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=group_id,
                    match=models.MatchValue(
                        value=gid,
                    ),
                )
            ]
        ),
        limit=response_limit,
    ).points
    print(search_result)
    return search_result


def main():
    if not collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=distance_metric
            ),
            hnsw_config=models.HnswConfigDiff(
                payload_m=16,
                m=0,
            ),
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name=group_id,
            field_schema=models.KeywordIndexParams(
                type="keyword",
                is_tenant=True,
            ),
        )

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=1,
                payload={group_id: "1"},
                vector=[0.9, 0.1, 0.1],
            ),
            models.PointStruct(
                id=2,
                payload={group_id: "1"},
                vector=[0.1, 0.9, 0.1],
            ),
            models.PointStruct(
                id=3,
                payload={group_id: "2"},
                vector=[0.1, 0.1, 0.9],
            ),
        ],
    )

    response1 = query_vector_in_group("1", [0.9, 0.1, 0.1])
    response2 = query_vector_in_group("2", [0.9, 0.1, 0.1])
    if response1[0].score > response2[0].score:
        print("Success!")
    else:
        print("Fail!")


if __name__ == "__main__":
    main()
