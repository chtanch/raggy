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


def query_multiple_kbs(kb_ids, query_vector) -> models.QueryResponse:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="kb_id",
                    match=models.MatchAny(any=kb_ids),
                ),
            ]
        ),
        limit=response_limit,
    ).points
    print(response)
    return response


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
        points=models.Batch(
            ids=[1, 2, 3],
            payloads=[
                {"kb_id": "kb_1"},
                {"kb_id": "kb_2"},
                {"kb_id": "kb_3"},
            ],
            vectors=[
                [0.9, 0.1, 0.1],
                [0.9, 0.1, 0.1],
                [0.9, 0.1, 0.1],
            ],
        ),
    )

    print(
        "TEST: Same vector value in multiple KBs. Must return the vector ids associated with the kb_ids being queried."
    )
    response = query_multiple_kbs(["kb_2", "kb_3"], [0.9, 0.1, 0.1])
    success = False
    for res in response:
        if res.id in (2, 3) and res.score >= 1.0:
            success = True
        else:
            success = False
            break

    if success:
        print("Success!")
    else:
        print("Fail!")


if __name__ == "__main__":
    main()
