from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pymilvus import MilvusClient

router = APIRouter(prefix="/milvus", tags=["milvus"])

MILVUS_URI = "http://192.168.17.30:19530"
COLLECTION = "demo_vectors"
DIM = 4

client: MilvusClient | None = None


def _get_client() -> MilvusClient:
    if client is None:
        raise HTTPException(status_code=503, detail="Milvus client not ready")
    return client


def startup() -> None:
    global client
    client = MilvusClient(uri=MILVUS_URI)

    if not client.has_collection(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            dimension=DIM,
            # metric_type options: L2 / IP / COSINE
        )


@router.get("/ping")
def milvus_ping() -> dict:
    cols = _get_client().list_collections()
    return {"ok": True, "collections": cols}


@router.post("/insert")
def milvus_insert() -> dict:
    data = [
        {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
        {"id": 2, "vector": [0.11, 0.19, 0.29, 0.41]},
    ]
    res = _get_client().insert(collection_name=COLLECTION, data=data)
    return {"insert_result": res}


@router.get("/search")
def milvus_search() -> dict:
    query_vec = [0.1, 0.2, 0.3, 0.4]
    res = _get_client().search(
        collection_name=COLLECTION,
        data=[query_vec],
        limit=2,
        output_fields=["id"],
    )
    return {"search_result": res}
