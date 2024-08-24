from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")

collection_name = "my_kb"
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    hnsw_config=models.HnswConfigDiff(
        payload_m=16,
        m=0,
    ),
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="kb_id",
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
            payload={"kb_id": "1"},
            vector=[0.9, 0.1, 0.1],
        ),
        models.PointStruct(
            id=2,
            payload={"kb_id": "1"},
            vector=[0.1, 0.9, 0.1],
        ),
        models.PointStruct(
            id=3,
            payload={"kb_id": "2"},
            vector=[0.1, 0.1, 0.9],
        ),
    ],
)

search_result = client.query_points(
    collection_name=collection_name,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="kb_id",
                match=models.MatchValue(
                    value="1",
                ),
            )
        ]
    ),
    limit=1,
).points
print(search_result)

search_result = client.query_points(
    collection_name=collection_name,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="kb_id",
                match=models.MatchValue(
                    value="2",
                ),
            )
        ]
    ),
    limit=1,
).points
print(search_result)
