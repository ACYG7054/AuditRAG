from __future__ import annotations

from fastapi import FastAPI
from pymilvus import MilvusClient

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

MILVUS_URI = "http://192.168.17.30:19530"  # 改成你 VM5 的 IP
COLLECTION = "demo_vectors"
DIM = 4

client: MilvusClient | None = None


@app.on_event("startup")
def startup():
    global client
    # 连接 Milvus（官方文档：默认端口 19530，使用 URI 连接）
    client = MilvusClient(uri=MILVUS_URI)

    # 若集合不存在则创建（用最小 schema：id + vector）
    if not client.has_collection(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            dimension=DIM,
            # metric_type 可选：L2 / IP / COSINE，默认通常 L2（不同版本可能略有差异）
        )


@app.get("/milvus/ping")
def milvus_ping():
    # 简单检查：列出集合
    cols = client.list_collections()
    return {"ok": True, "collections": cols}


@app.post("/milvus/insert")
def milvus_insert():
    # 插入两条向量
    data = [
        {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
        {"id": 2, "vector": [0.11, 0.19, 0.29, 0.41]},
    ]
    res = client.insert(collection_name=COLLECTION, data=data)
    return {"insert_result": res}


@app.get("/milvus/search")
def milvus_search():
    # 搜索与 query 最相近的 top2
    query_vec = [0.1, 0.2, 0.3, 0.4]
    res = client.search(
        collection_name=COLLECTION,
        data=[query_vec],
        limit=2,
        output_fields=["id"],
    )
    return {"search_result": res}
'''
功能一：
快速复用历史问题

1.做一个导入接口，从RocketMQ去拉取消息嵌入Milvus向量库，批量插入审计问题
考虑好Collection的id、向量和元数据

2.做一个检索接口，用户输入一段话，根据用户的内容去做混合检索
输入预处理，去噪
向量检索
业务过滤
重排序
只返回 10 条低于阈值的剔除


'''


'''
智能底稿导入
用户输入一段文件，根据用户的信息去整理，并填写好表单

1. POST /manuscript/import                                                                                                                                                                             
     上传文件 + 项目/审计单位等上下文，返回 jobId                                                                                                                                                        
2. GET /manuscript/import/{jobId}/status                                                                                                                                                               
     返回解析进度、是否完成                                                                                                                                                                              
3. GET /manuscript/import/{jobId}/preview                                                                                                                                                              
     返回结构化结果 + 置信度 + 异常                                                                                                                                                                      
4. POST /manuscript/import/{jobId}/confirm                                                                                                                                                             
     用户确认/修正后入库                                                                                                                                                                                 
5. POST /manuscript/import/{jobId}/cancel                                                                                                                                                              
     终止导入或清理临时数据 

'''



