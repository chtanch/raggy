from qdrant_client import QdrantClient, models

# VectorDB parameters
location = ":memory:"
collection_name = "my_kb"
vector_size = 3
distance_metric = models.Distance.COSINE
group_id = "kb_id"
response_limit = 1

client = QdrantClient(location=location)


def query_vector_in_group(gid, query_vector):
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


def main():
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance_metric),
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

    query_vector_in_group("1", [0.9, 0.1, 0.1])
    query_vector_in_group("2", [0.9, 0.1, 0.1])


if __name__ == "__main__":
    main()
