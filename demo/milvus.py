from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pymilvus import MilvusClient

from config import COLLECTION, DIM, MILVUS_URI

# Milvus 示例接口
router = APIRouter(prefix="/milvus", tags=["milvus"])

client: MilvusClient | None = None


def _get_client() -> MilvusClient:
    # 客户端未就绪时返回 503
    if client is None:
        raise HTTPException(status_code=503, detail="Milvus client not ready")
    return client


def startup() -> None:
    # 初始化 Milvus 客户端与集合
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
    # 健康检查：返回当前集合列表
    cols = _get_client().list_collections()
    return {"ok": True, "collections": cols}


@router.post("/insert")
def milvus_insert() -> dict:
    # 插入示例向量数据
    data = [
        {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
        {"id": 2, "vector": [0.11, 0.19, 0.29, 0.41]},
    ]
    res = _get_client().insert(collection_name=COLLECTION, data=data)
    return {"insert_result": res}


@router.get("/search")
def milvus_search() -> dict:
    # 基于示例向量的近邻检索
    query_vec = [0.1, 0.2, 0.3, 0.4]
    res = _get_client().search(
        collection_name=COLLECTION,
        data=[query_vec],
        limit=2,
        output_fields=["id"],
    )
    return {"search_result": res}
