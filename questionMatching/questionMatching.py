'''
功能一：
快速复用历史问题

1.做一个导入接口，从RocketMQ去拉取消息嵌入Milvus向量库，批量插入审计问题

2.做一个检索接口，用户输入一段话，根据用户的内容去做混合检索
输入预处理，去噪
混合检索
重排序
只返回 10 条低于阈值的剔除
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import importlib
from pathlib import Path
import threading
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pymilvus import DataType, Function, FunctionType, MilvusClient
from pymilvus.milvus_client.index import IndexParams

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.vector_stores.milvus import IndexManagement, MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction

import config as app_config
from config import (
    AUDIT_COLLECTION,
    AUDIT_DENSE_DIM,
    DASHSCOPE_API_KEY,
    DASHSCOPE_EMBED_MODEL,
    MILVUS_URI,
)

# 审计问题匹配：导入与检索接口
router = APIRouter(prefix="/question-matching", tags=["question-matching"])
# Milvus 字段与集合配置
COLLECTION_NAME = AUDIT_COLLECTION
DENSE_DIM = AUDIT_DENSE_DIM
DENSE_FIELD = "dense_vec"
SPARSE_FIELD = "sparse_vec"
TEXT_FIELD = "text_all"
SCALAR_FIELDS = [
    "source_id",
    "problem_type",
    "problem_nature",
    "problem_desc",
    "basis",
    "advice",
]
SCALAR_FIELD_TYPES = [DataType.VARCHAR] * len(SCALAR_FIELDS)
OUTPUT_FIELDS = [TEXT_FIELD] + SCALAR_FIELDS
client: MilvusClient | None = None
_index: VectorStoreIndex | None = None
# 字段权重配置键（用于重排序层）
FIELD_WEIGHT_KEYS = (
    "problem_type",
    "problem_nature",
    "problem_desc",
    "basis",
    "advice",
)
# 字段权重默认值（Mock Nacos 无配置或配置非法时使用）
DEFAULT_FIELD_WEIGHTS: Dict[str, float] = {
    "problem_type": 0.1,
    "problem_nature": 0.1,
    "problem_desc": 0.35,
    "basis": 0.25,
    "advice": 0.2,
}
# 字段权重缓存锁，防止并发读写冲突
_field_weight_lock = threading.Lock()
# 字段权重内存缓存，支持热更新
_field_weight_config: Dict[str, float] = DEFAULT_FIELD_WEIGHTS.copy()
# Nacos 监听器单例（Mock 版本）
_nacos_listener: "NacosConfigListener | None" = None

# 导入记录模型（来自消息或批量接口）
# 导入记录模型（单条审计问题的数据结构）
class ImportRecord(BaseModel):
    source_id: str = Field(..., min_length=1, max_length=64)
    problem_type: str = Field(..., min_length=1, max_length=64)
    problem_nature: str = Field(..., min_length=1, max_length=64)
    problem_desc: str = Field(..., min_length=1, max_length=1024)
    basis: str = Field(..., min_length=1, max_length=1024)
    advice: str = Field(..., min_length=1, max_length=1024)

# 导入请求（可带批量记录或按 batch_size 拉取）
# 导入请求模型（支持批量记录或按 batch_size 拉取）
class ImportRequest(BaseModel):
    batch_size: int = Field(100, ge=1, le=5000)
    records: Optional[List[ImportRecord]] = None


# 检索请求参数
# 检索请求模型（混合检索与重排控制参数）
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)
    min_score: float = Field(0.0)
    dense_weight: float = Field(0.7, gt=0.0)
    sparse_weight: float = Field(0.3, gt=0.0)
    use_rrf: bool = False
    filter_source_id: Optional[str] = None
    filter_problem_type: Optional[str] = None
    filter_problem_nature: Optional[str] = None

# 获取 LlamaIndex 索引实例，未初始化时抛 503
def _get_index() -> VectorStoreIndex:
    # 索引未就绪时返回 503，避免空指针
    if _index is None:
        raise HTTPException(status_code=503, detail="LlamaIndex not ready")
    return _index



# 构建元数据过滤条件（按 source_id / problem_type / problem_nature）
def _build_metadata_filters(req: SearchRequest) -> Optional[MetadataFilters]:
    # 根据请求条件构建元数据过滤器
    filters = []
    if req.filter_source_id:
        filters.append(ExactMatchFilter(key="source_id", value=req.filter_source_id))
    if req.filter_problem_type:
        filters.append(ExactMatchFilter(key="problem_type", value=req.filter_problem_type))
    if req.filter_problem_nature:
        filters.append(ExactMatchFilter(key="problem_nature", value=req.filter_problem_nature))
    if not filters:
        return None
    return MetadataFilters(filters=filters)

# 规范化文本（清理首尾空白与多余空格）
def _normalize_text(text: str) -> str:
    # 规整空白字符，避免空检索
    return " ".join(text.strip().split())


def _get_config_path() -> Path:
    # 读取根目录 config.py，作为 Mock Nacos 配置来源
    return Path(__file__).resolve().parents[1] / "config.py"


def _normalize_weight_config(raw_config: Dict[str, Any]) -> Dict[str, float]:
    # 清洗并校验配置：缺失/非法值回退默认权重
    if not isinstance(raw_config, dict):
        return DEFAULT_FIELD_WEIGHTS.copy()
    raw_weights = raw_config.get("weights", {})
    normalized: Dict[str, float] = {}
    for key in FIELD_WEIGHT_KEYS:
        value = raw_weights.get(key, DEFAULT_FIELD_WEIGHTS[key])
        try:
            weight = float(value)
        except (TypeError, ValueError):
            weight = DEFAULT_FIELD_WEIGHTS[key]
        if weight < 0.0:
            weight = 0.0
        normalized[key] = weight
    return normalized


def _refresh_field_weights(raw_config: Dict[str, Any]) -> None:
    # 刷新内存中的权重配置（线程安全）
    normalized = _normalize_weight_config(raw_config)
    with _field_weight_lock:
        _field_weight_config.update(normalized)


def _get_field_weights() -> Dict[str, float]:
    # 对外返回权重副本，避免被外部修改
    with _field_weight_lock:
        return dict(_field_weight_config)


class NacosConfigListener:
    def __init__(self, config_path: Path, poll_interval: float) -> None:
        # 监听配置文件变更（Mock Nacos），支持轮询间隔配置
        self._config_path = config_path
        self._poll_interval = max(float(poll_interval), 0.5)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_mtime: Optional[float] = None

    def start(self) -> None:
        # 启动后台线程，避免重复启动
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="nacos-config-listener",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        # 首次强制加载配置，后续按间隔轮询
        self._reload_if_needed(force=True)
        while not self._stop_event.is_set():
            time.sleep(self._poll_interval)
            self._reload_if_needed(force=False)

    def _reload_if_needed(self, force: bool) -> None:
        # 文件变更检测：mtime 变化才触发 reload
        try:
            mtime = self._config_path.stat().st_mtime
        except OSError:
            return
        if not force and self._last_mtime is not None and mtime <= self._last_mtime:
            return
        self._last_mtime = mtime
        try:
            importlib.reload(app_config)
        except Exception:
            return
        raw_config = getattr(app_config, "NACOS_FIELD_WEIGHT_CONFIG", None)
        if not isinstance(raw_config, dict):
            return
        poll_interval = raw_config.get("poll_interval")
        if isinstance(poll_interval, (int, float)) and poll_interval > 0:
            self._poll_interval = max(float(poll_interval), 0.5)
        _refresh_field_weights(raw_config)


def _start_nacos_listener() -> None:
    # 启动 Mock Nacos 监听器，确保只启动一次
    global _nacos_listener
    if _nacos_listener is not None:
        _nacos_listener.start()
        return
    raw_config = getattr(app_config, "NACOS_FIELD_WEIGHT_CONFIG", {})
    poll_interval = 2.0
    if isinstance(raw_config, dict):
        raw_interval = raw_config.get("poll_interval")
        if isinstance(raw_interval, (int, float)) and raw_interval > 0:
            poll_interval = raw_interval
        _refresh_field_weights(raw_config)
    _nacos_listener = NacosConfigListener(_get_config_path(), poll_interval)
    _nacos_listener.start()


def _to_char_bigrams(text: str) -> List[str]:
    # 字符二元组切分：兼容中文与英文，无需额外分词依赖
    if not text:
        return []
    if len(text) == 1:
        return [text]
    return [text[i : i + 2] for i in range(len(text) - 1)]


def _field_similarity(query_text: str, field_text: str) -> float:
    # 基于字符二元组的 Jaccard 相似度，用于字段匹配打分
    query = _normalize_text(query_text).replace(" ", "")
    field = _normalize_text(field_text).replace(" ", "")
    if not query or not field:
        return 0.0
    query_grams = _to_char_bigrams(query)
    field_grams = _to_char_bigrams(field)
    if not query_grams or not field_grams:
        return 0.0
    query_set = set(query_grams)
    field_set = set(field_grams)
    union = query_set | field_set
    if not union:
        return 0.0
    return len(query_set & field_set) / len(union)


def _rerank_nodes(nodes: List[Any], query_text: str) -> List[Any]:
    # 根据字段权重对候选结果重排序
    if not nodes:
        return nodes
    weights = _get_field_weights()
    if not weights:
        return nodes
    query = _normalize_text(query_text)
    if not query:
        return nodes
    scored: List[Tuple[float, Any]] = []
    for item in nodes:
        # base_score 为检索得分，extra_score 为字段权重加成
        base_score = getattr(item, "score", 0.0) or 0.0
        node = getattr(item, "node", None)
        metadata = getattr(node, "metadata", {}) if node else {}
        extra_score = 0.0
        for field_name, weight in weights.items():
            if weight <= 0.0:
                continue
            field_value = metadata.get(field_name, "")
            extra_score += weight * _field_similarity(query, str(field_value))
        rerank_score = base_score + extra_score
        try:
            # 写回 score，便于后续统一使用
            item.score = rerank_score
        except Exception:
            pass
        scored.append((rerank_score, item))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored]


# 拼接全文检索字段（用于稠密/稀疏混合检索）
def _build_text_all(record: ImportRecord) -> str:
    # 拼接用于检索的全文字段
    return "\n".join([record.problem_desc, record.basis, record.advice])


# Mock RocketMQ 拉取数据（本地调试用）
def _mock_pull_from_rocketmq(batch_size: int) -> List[ImportRecord]:
    # 模拟从 RocketMQ 拉取数据（本地调试用）
    now = int(time.time())
    records: List[ImportRecord] = []
    for i in range(batch_size):
        records.append(
            ImportRecord(
                source_id=f"mock-{now}-{i}",
                problem_type="mock_type",
                problem_nature="mock_nature",
                problem_desc=f"mock problem desc {i}",
                basis=f"mock basis {i}",
                advice=f"mock advice {i}",
            )
        )
    return records


# 确保 Milvus 集合与索引存在
def _ensure_collection() -> None:
    # 确保集合与索引存在
    milvus = _get_client()
    # 集合已存在则直接加载
    if milvus.has_collection(COLLECTION_NAME):
        milvus.load_collection(COLLECTION_NAME)
        return

    # 初始化集合 Schema
    # 创建集合 Schema
    schema = milvus.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
        description="audit problem matching",
    )
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    schema.add_field(
        field_name="source_id",
        datatype=DataType.VARCHAR,
        max_length=64,
        nullable=False,
    )
    schema.add_field(
        field_name="problem_type",
        datatype=DataType.VARCHAR,
        max_length=64,
        nullable=False,
    )
    schema.add_field(
        field_name="problem_nature",
        datatype=DataType.VARCHAR,
        max_length=64,
        nullable=False,
    )
    schema.add_field(
        field_name="problem_desc",
        datatype=DataType.VARCHAR,
        max_length=1024,
        nullable=False,
    )
    schema.add_field(
        field_name="basis",
        datatype=DataType.VARCHAR,
        max_length=1024,
        nullable=False,
    )
    schema.add_field(
        field_name="advice",
        datatype=DataType.VARCHAR,
        max_length=1024,
        nullable=False,
    )
    schema.add_field(
        field_name=TEXT_FIELD,
        datatype=DataType.VARCHAR,
        max_length=4096,
        nullable=False,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
        enable_match=True,
    )
    schema.add_field(
        field_name=DENSE_FIELD,
        datatype=DataType.FLOAT_VECTOR,
        dim=DENSE_DIM,
        nullable=False,
    )
    schema.add_field(
        field_name=SPARSE_FIELD,
        datatype=DataType.SPARSE_FLOAT_VECTOR,
        nullable=False,
    )
    schema.add_function(
        Function(
            name="bm25_func1",
            function_type=FunctionType.BM25,
            input_field_names=[TEXT_FIELD],
            output_field_names=[SPARSE_FIELD],
        )
    )

    # 构建向量索引
    index_params = IndexParams()
    index_params.add_index(
        field_name=DENSE_FIELD,
        index_name=DENSE_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
        params={"mmap.enabled": "false"},
    )
    index_params.add_index(
        field_name=SPARSE_FIELD,
        index_name=SPARSE_FIELD,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    # 创建集合并设置索引参数
    milvus.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Bounded",
        shards_num=1,
        properties={"timezone": "Asia/Shanghai"},
    )
    # 创建后立即加载集合
    milvus.load_collection(COLLECTION_NAME)


# 获取 Milvus 客户端实例，未初始化时抛 503
def _get_client() -> MilvusClient:
    # Milvus 客户端未就绪时返回 503
    if client is None:
        raise HTTPException(status_code=503, detail="Milvus client not ready")
    return client


# 构建 DashScope 向量嵌入模型
def _build_embed_model() -> DashScopeEmbedding:
    # DashScope 嵌入模型初始化
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY ???")
    return DashScopeEmbedding(api_key=DASHSCOPE_API_KEY, model_name=DASHSCOPE_EMBED_MODEL)


# 构建 Milvus 向量存储（支持稠密/稀疏混合检索）
def _build_vector_store(
    hybrid_ranker: str,
    hybrid_ranker_params: Optional[Dict[str, Any]],
) -> MilvusVectorStore:
    # 构建支持稀疏+稠密的向量存储
    bm25_function = BM25BuiltInFunction(
        input_field_names=TEXT_FIELD,
        output_field_names=SPARSE_FIELD,
        analyzer_params={"type": "chinese"},
        enable_match=True,
    )
    index_management = getattr(
        IndexManagement,
        "NO_VALIDATION",
        IndexManagement.CREATE_IF_NOT_EXISTS,
    )
    return MilvusVectorStore(
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        dim=DENSE_DIM,
        text_key=TEXT_FIELD,
        embedding_field=DENSE_FIELD,
        enable_sparse=True,
        sparse_embedding_field=SPARSE_FIELD,
        sparse_embedding_function=bm25_function,
        scalar_field_names=SCALAR_FIELDS,
        scalar_field_types=SCALAR_FIELD_TYPES,
        output_fields=OUTPUT_FIELDS,
        similarity_metric="COSINE",
        consistency_level="Bounded",
        index_management=index_management,
        hybrid_ranker=hybrid_ranker,
        hybrid_ranker_params=hybrid_ranker_params,
    )


# 构建默认索引（加权混合检索）
def _build_default_index() -> VectorStoreIndex:
    # 默认使用加权混合检索
    vector_store = _build_vector_store(
        hybrid_ranker="WeightedRanker",
        hybrid_ranker_params={"weights": [0.7, 0.3]},
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


# 启动初始化：Milvus、集合、嵌入模型与索引
def startup() -> None:
    # 初始化 Milvus、集合与 LlamaIndex
    global client, _index
    client = MilvusClient(uri=MILVUS_URI)
    _ensure_collection()
    Settings.embed_model = _build_embed_model()
    _index = _build_default_index()
    # 启动配置监听，支持字段权重热更新
    _start_nacos_listener()


# 导入审计问题数据（支持批量或 Mock 拉取）
@router.post("/import")
def import_from_rocketmq(req: ImportRequest) -> Dict[str, Any]:
    """导入审计问题（支持批量或模拟拉取）"""

    # 如果没初始化，那么先初始化
    if client is None or _index is None:
        startup()

    # 获取导入数据来源（请求传入或 Mock 拉取）
    records = req.records if req.records else _mock_pull_from_rocketmq(req.batch_size)
    nodes: List[TextNode] = []
    # 构建节点与元数据
    for record in records:
        text_all = _build_text_all(record)
        metadata = {
            "source_id": record.source_id,
            "problem_type": record.problem_type,
            "problem_nature": record.problem_nature,
            "problem_desc": record.problem_desc,
            "basis": record.basis,
            "advice": record.advice,
        }
        nodes.append(TextNode(text=text_all, id_=record.source_id, metadata=metadata))

    if not nodes:
        return {"ok": True, "inserted": 0, "collection": COLLECTION_NAME}

    _get_index().insert_nodes(nodes)
    return {
        "ok": True,
        "inserted": len(nodes),
        "collection": COLLECTION_NAME,
    }


# 混合检索入口（稠密 + 稀疏 + 重排）
@router.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    """混合检索（稠密 + 稀疏），必要时回退默认检索"""
    if client is None or _index is None:
        startup()

    query_text = _normalize_text(req.query)
    if not query_text:
        raise HTTPException(status_code=400, detail="query is empty")

    filters = _build_metadata_filters(req)

    # 根据开关选择融合策略（RRF 或加权）
    if req.use_rrf:
        hybrid_ranker = "RRFRanker"
        hybrid_ranker_params = {"k": 60}
    else:
        hybrid_ranker = "WeightedRanker"
        hybrid_ranker_params = {"weights": [req.dense_weight, req.sparse_weight]}

    vector_store = _build_vector_store(
        hybrid_ranker=hybrid_ranker,
        hybrid_ranker_params=hybrid_ranker_params,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    # 优先尝试混合检索，失败则回退默认检索
    try:
        retriever = index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=req.top_k,
            filters=filters,
        )
        nodes = retriever.retrieve(query_text)
        # 字段权重重排（基于 Mock Nacos 配置）
        nodes = _rerank_nodes(nodes, query_text)
        results = _format_nodes(nodes, req.min_score)
        return {"ok": True, "count": len(results), "results": results, "hybrid": True}
    except Exception as exc:  # pragma: no cover - fallback for schema/query mismatch
        retriever = index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
            similarity_top_k=req.top_k,
            filters=filters,
        )
        nodes = retriever.retrieve(query_text)
        # 字段权重重排（基于 Mock Nacos 配置）
        nodes = _rerank_nodes(nodes, query_text)
        results = _format_nodes(nodes, req.min_score)
        return {
            "ok": True,
            "count": len(results),
            "results": results,
            "hybrid": False,
            "fallback_reason": str(exc),
        }


# 格式化检索结果，并按 min_score 过滤
def _format_nodes(
    nodes: List[Any],
    min_score: float,
) -> List[Dict[str, Any]]:
    # 过滤低分节点并提取需要的字段
    results: List[Dict[str, Any]] = []
    for item in nodes:
        score = getattr(item, "score", None)
        if score is None:
            continue

        if score < min_score:
            continue
        node = getattr(item, "node", None)
        if node is None:
            continue
        metadata = getattr(node, "metadata", {}) or {}
        results.append(
            {
                "id": getattr(node, "node_id", None),
                "score": score,
                "source_id": metadata.get("source_id"),
                "problem_type": metadata.get("problem_type"),
                "problem_nature": metadata.get("problem_nature"),
                "problem_desc": metadata.get("problem_desc"),
                "basis": metadata.get("basis"),
                "advice": metadata.get("advice"),
            }
        )
    return results
