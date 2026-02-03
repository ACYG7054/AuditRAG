MILVUS_URI: str = "http://192.168.17.30:19530"
COLLECTION: str = "demo_vectors"
DIM: int = 4

AUDIT_COLLECTION: str = "audit_problem_rag"
AUDIT_DENSE_DIM: int = 1024

# DashScope 向量模型配置（占位，稍后填写）
DASHSCOPE_API_KEY: str = "sk-42007e4d1f4a477eaa401820f4a85a13"

#embedding模型用阿里云text-embedding-v4  默认 1024
DASHSCOPE_EMBED_MODEL: str = "qwen3-vl-embedding"

#rerank模型用阿里云的XXX，先用qwen3-vl-rerank混混


# Mock Nacos config for rerank field weights (hot-reload via file change).
NACOS_FIELD_WEIGHT_CONFIG = {
    "data_id": "audit_rerank_weights",
    "group": "DEFAULT_GROUP",
    "namespace": "public",
    "poll_interval": 2.0,
    "weights": {
        "problem_type": 0.1,
        "problem_nature": 0.1,
        "problem_desc": 0.35,
        "basis": 0.25,
        "advice": 0.2,
    },
}
