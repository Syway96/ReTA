#!/usr/bin/env python3
"""
智能问答系统
"""

import os
import sys
import json
import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ==================== 统一配置管理 ====================

@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    local_path: Optional[str] = None
    online_fallback: Optional[str] = None
    device: Optional[str] = None
    normalize_embeddings: Optional[bool] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """大语言模型配置"""
    provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: Optional[float] = None
    num_predict: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_ctx: Optional[int] = None


@dataclass
class RetrievalConfig:
    """检索配置"""
    k_per_store: Optional[int] = None
    total_max_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    enable_reranking: Optional[bool] = None
    dynamic_complexity: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankConfig:
    """重排序配置"""
    enabled: Optional[bool] = None
    method: Optional[str] = None
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None
    use_cross_encoder: Optional[bool] = None
    cross_encoder_model: Optional[str] = None
    cross_encoder_local_path: Optional[str] = None  # 本地模型路径


@dataclass
class PromptConfig:
    """提示词配置"""
    system_template: Optional[str] = None
    human_template: Optional[str] = None


@dataclass
class SystemConfig:
    """系统配置"""
    enable_multiple_stores: Optional[bool] = None
    show_retrieval_info: Optional[bool] = None
    log_level: Optional[str] = None
    streaming_output: Optional[bool] = None
    streaming_delay: Optional[float] = None


@dataclass
class QASystemConfig:
    """问答系统配置"""
    project_dir: Optional[str] = None
    vector_store_dir: Optional[str] = None
    data_dir: Optional[str] = None
    embedding: Optional[EmbeddingConfig] = None
    llm: Optional[LLMConfig] = None
    retrieval: Optional[RetrievalConfig] = None
    rerank: Optional[RerankConfig] = None
    prompt: Optional[PromptConfig] = None
    system: Optional[SystemConfig] = None


class UnifiedConfigManager:
    """统一配置管理器"""

    @staticmethod
    def load_config(config_path: str = "config.yaml") -> QASystemConfig:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            # 提取配置
            qa = config.get('qa_system', {})
            vector = config.get('vector_processing', {})
            data = config.get('data_processing', {})

            # 构建配置数据 - 使用dataclass.from_dict处理嵌套配置
            from dacite import from_dict
            
            # 构建完整的配置字典
            config_dict = {}
            config_dict.update(qa)
            config_dict['data_dir'] = data.get("output_dir")
            config_dict['embedding'] = vector.get("embedding")
            
            return from_dict(QASystemConfig, config_dict)

        except Exception as e:
            logging.error(f"配置加载失败: {e}")
            return QASystemConfig()


class DocumentReranker:
    """文档重排序器 - 支持多种重排序策略"""

    def __init__(self, config: RerankConfig, embedding_model=None):
        """初始化重排序器"""
        self.config = config
        self.embedding_model = embedding_model
        self.cross_encoder = None

        if config.use_cross_encoder and config.cross_encoder_model:
            self._init_cross_encoder()

    def _init_cross_encoder(self):
        """初始化交叉编码器"""
        try:
            from sentence_transformers import CrossEncoder

            # 优先使用本地路径
            if self.config.cross_encoder_local_path:
                model_path = self.config.cross_encoder_local_path
                logging.info(f"📂 使用本地模型路径: {model_path}")
            else:
                model_path = self.config.cross_encoder_model
                logging.info(f"🌐 使用在线模型: {model_path}")

            self.cross_encoder = CrossEncoder(model_path)
            logging.info(f"✅ 交叉编码器加载成功: {model_path}")
        except Exception as e:
            logging.warning(f"交叉编码器加载失败: {e}")
            logging.info("将使用基于嵌入的重排序方法")

    def rerank(self, query: str, documents: List[Any]) -> List[Any]:
        """对检索结果进行重排序"""
        if not documents:
            return documents

        if not self.config.enabled:
            logging.debug("重排序未启用")
            return documents

        logging.info(f"🔄 开始重排序，方法: {self.config.method}，文档数: {len(documents)}")

        try:
            if self.config.method == "cross_encoder" and self.cross_encoder:
                return self._rerank_with_cross_encoder(query, documents)
            elif self.config.method == "embedding_similarity":
                return self._rerank_with_embedding_similarity(query, documents)
            elif self.config.method == "hybrid":
                return self._rerank_hybrid(query, documents)
            elif self.config.method == "keyword_boost":
                return self._rerank_with_keyword_boost(query, documents)
            else:
                logging.warning(f"未知的重排序方法: {self.config.method}，使用默认方法")
                return self._rerank_with_embedding_similarity(query, documents)

        except Exception as e:
            logging.error(f"重排序失败: {e}")
            return documents

    def _rerank_with_cross_encoder(self, query: str, documents: List[Any]) -> List[Any]:
        """使用交叉编码器重排序"""
        if not self.cross_encoder:
            return self._rerank_with_embedding_similarity(query, documents)

        # 准备查询-文档对
        query_doc_pairs = [(query, doc.page_content) for doc in documents]

        # 计算相关性分数
        scores = self.cross_encoder.predict(query_doc_pairs)

        # 更新文档分数
        for doc, score in zip(documents, scores):
            doc.metadata['rerank_score'] = float(score)

        # 按重排序分数排序（降序）
        reranked = sorted(documents, key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)

        # 应用阈值
        if self.config.score_threshold > 0:
            reranked = [
                doc for doc in reranked
                if doc.metadata.get('rerank_score', 0) >= self.config.score_threshold
            ]

        # 限制数量
        top_k = self.config.top_k if self.config.top_k else len(reranked)
        return reranked[:top_k]

    def _rerank_with_embedding_similarity(self, query: str, documents: List[Any]) -> List[Any]:
        """使用嵌入相似度重排序"""
        if not self.embedding_model:
            return documents

        try:
            # 计算查询的嵌入
            query_embedding = self.embedding_model.embed_query(query)

            # 计算每个文档的嵌入
            doc_embeddings = [
                self.embedding_model.embed_query(doc.page_content)
                for doc in documents
            ]

            # 计算余弦相似度
            import numpy as np
            query_vec = np.array(query_embedding)
            similarities = []

            for doc_vec in doc_embeddings:
                doc_vec = np.array(doc_vec)
                similarity = float(np.dot(query_vec, doc_vec))
                similarities.append(similarity)

            # 更新文档分数
            for doc, similarity in zip(documents, similarities):
                doc.metadata['rerank_score'] = similarity

            # 按重排序分数排序（降序）
            reranked = sorted(documents, key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)

            # 应用阈值
            if self.config.score_threshold > 0:
                reranked = [
                    doc for doc in reranked
                    if doc.metadata.get('rerank_score', 0) >= self.config.score_threshold
                ]

            # 限制数量
            top_k = self.config.top_k if self.config.top_k else len(reranked)
            return reranked[:top_k]

        except Exception as e:
            logging.error(f"嵌入相似度重排序失败: {e}")
            return documents

    def _rerank_hybrid(self, query: str, documents: List[Any]) -> List[Any]:
        """混合重排序方法"""
        # 获取原始相似度分数
        original_scores = [doc.metadata.get('similarity_score', 0) for doc in documents]

        # 计算重排序分数
        if self.cross_encoder:
            query_doc_pairs = [(query, doc.page_content) for doc in documents]
            rerank_scores = self.cross_encoder.predict(query_doc_pairs)
        elif self.embedding_model:
            import numpy as np
            query_embedding = self.embedding_model.embed_query(query)
            query_vec = np.array(query_embedding)
            rerank_scores = []
            for doc in documents:
                doc_vec = np.array(self.embedding_model.embed_query(doc.page_content))
                rerank_scores.append(float(np.dot(query_vec, doc_vec)))
        else:
            return documents

        # 混合分数（加权平均）
        alpha = 0.6  # 重排序分数权重
        beta = 0.4   # 原始分数权重

        hybrid_scores = []
        for orig_score, rerank_score in zip(original_scores, rerank_scores):
            # 归一化分数
            norm_orig = (orig_score - min(original_scores)) / (max(original_scores) - min(original_scores) + 1e-6)
            norm_rerank = (rerank_score - min(rerank_scores)) / (max(rerank_scores) - min(rerank_scores) + 1e-6)
            hybrid_score = alpha * norm_rerank + beta * norm_orig
            hybrid_scores.append(hybrid_score)

        # 更新文档分数
        for doc, score in zip(documents, hybrid_scores):
            doc.metadata['rerank_score'] = score

        # 按混合分数排序
        reranked = sorted(documents, key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)

        # 应用阈值
        if self.config.score_threshold > 0:
            reranked = [
                doc for doc in reranked
                if doc.metadata.get('rerank_score', 0) >= self.config.score_threshold
            ]

        # 限制数量
        top_k = self.config.top_k if self.config.top_k else len(reranked)
        return reranked[:top_k]

    def _rerank_with_keyword_boost(self, query: str, documents: List[Any]) -> List[Any]:
        """基于关键词提升的重排序"""
        import re
        from collections import Counter

        # 提取查询关键词
        keywords = re.findall(r'\w+', query.lower())
        keyword_counter = Counter(keywords)

        # 计算每个文档的关键词匹配分数
        for doc in documents:
            content = doc.page_content.lower()
            keyword_score = 0

            for keyword, count in keyword_counter.items():
                keyword_count = content.count(keyword)
                keyword_score += keyword_count * count

            # 原始相似度分数
            original_score = doc.metadata.get('similarity_score', 0)

            # 混合分数
            boost_factor = 0.1
            rerank_score = original_score * (1 + boost_factor * keyword_score / len(keywords))

            doc.metadata['rerank_score'] = rerank_score

        # 按重排序分数排序
        reranked = sorted(documents, key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)

        # 限制数量
        top_k = self.config.top_k if self.config.top_k else len(reranked)
        return reranked[:top_k]


class VectorStoreManager:
    """兼容版向量库管理器 - 能够读取第二阶段生成的向量库"""

    def __init__(self, config: QASystemConfig):
        self.config = config
        self.embedding = None
        self.vector_stores = {}
        self.store_info = {}
        self.reranker = None

        self._setup_logging()
        self._init_embedding_model()
        self._init_reranker()
        self._load_vector_stores()

    def _init_reranker(self):
        """初始化重排序器"""
        if self.config.rerank and self.config.rerank.enabled:
            try:
                self.reranker = DocumentReranker(
                    config=self.config.rerank,
                    embedding_model=self.embedding
                )
                logging.info(f"✅ 重排序器初始化成功: {self.config.rerank.method}")
            except Exception as e:
                logging.warning(f"重排序器初始化失败: {e}")
                self.reranker = None

    def _setup_logging(self):
        """设置日志"""
        # 安全地获取日志级别，如果配置有问题则使用默认值
        log_level = "INFO"  # 默认值
        if self.config.system and hasattr(self.config.system, 'log_level') and self.config.system.log_level:
            log_level = self.config.system.log_level
        
        # 确保日志目录存在
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/vector_manager.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            model_path = self.config.embedding.local_path

            # 检查本地模型是否存在
            if not os.path.exists(model_path):
                logging.warning(f"本地模型路径不存在: {model_path}")
                logging.info(f"使用在线模型: {self.config.embedding.online_fallback}")
                model_path = self.config.embedding.online_fallback

            # 使用第二阶段的参数
            self.embedding = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs=self.config.embedding.model_kwargs,
                encode_kwargs=self.config.embedding.encode_kwargs
            )

            logging.info(f"✅ 嵌入模型初始化完成: {os.path.basename(model_path)}")

        except Exception as e:
            logging.error(f"嵌入模型初始化失败: {e}")
            raise

    def _load_vector_stores(self):
        """加载所有向量库"""
        vector_store_dir = Path(self.config.vector_store_dir)

        if not vector_store_dir.exists():
            logging.error(f"向量库目录不存在: {vector_store_dir}")
            logging.info(f"请先运行第二阶段代码创建向量库")
            return

        # 查找第二阶段生成的向量库
        store_paths = []
        for item in vector_store_dir.iterdir():
            if item.is_dir() and item.name.startswith("kb_"):
                # 检查是否包含Chroma文件
                chroma_files = list(item.glob("*.parquet")) + list(item.glob("*.sqlite3"))
                if chroma_files:
                    store_paths.append(item)

        if not store_paths:
            logging.error(f"在 {vector_store_dir} 中未找到有效的向量库")
            logging.info(f"请确保已经运行: python vector_store.py --process-all")
            return

        logging.info(f"找到 {len(store_paths)} 个向量库")

        # 并行加载向量库
        with ThreadPoolExecutor(max_workers=min(4, len(store_paths))) as executor:
            futures = {
                executor.submit(self._load_single_store, store_path): store_path
                for store_path in store_paths
            }

            for future in futures:
                store_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        store_id, vector_store, store_info = result
                        self.vector_stores[store_id] = vector_store
                        self.store_info[store_id] = store_info
                        logging.info(f"✅ 加载向量库: {store_id} ({store_info.get('num_docs', 0)}个文档)")
                except Exception as e:
                    logging.error(f"加载向量库失败 {store_path.name}: {e}")

        if self.vector_stores:
            logging.info(f"✅ 成功加载 {len(self.vector_stores)} 个向量库")
        else:
            logging.error("❌ 未能加载任何向量库")

    def _load_single_store(self, store_path: Path) -> Optional[Tuple[str, Any, Dict]]:
        """加载单个向量库"""
        try:
            store_id = store_path.name

            # 从store_id提取原始data_id（移除kb_前缀）
            if store_id.startswith("kb_"):
                data_id = store_id[3:]  # 移除"kb_"前缀
            else:
                data_id = store_id

            # 确定集合名称
            collection_name = store_id

            # 尝试加载向量库
            from langchain_chroma import Chroma

            try:
                vector_store = Chroma(
                    persist_directory=str(store_path),
                    embedding_function=self.embedding,
                    collection_name=collection_name
                )

                # 测试读取
                test_count = vector_store._collection.count()

                if test_count > 0:
                    logging.debug(f"  成功加载 {store_id}: {test_count} 个文档")

                    # 向量库信息
                    store_info = {"num_docs": test_count, "path": str(store_path)}

                    # 读取对应的数据文件信息
                    data_file = Path(self.config.data_dir) / f"{data_id}.json"
                    if data_file.exists():
                        try:
                            with open(data_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            store_info["source_file"] = str(data_file)
                            store_info["source_chunks"] = len(data)

                            # 从第一个文档获取原始PDF信息
                            if data and len(data) > 0:
                                first_doc = data[0]
                                metadata = first_doc.get("metadata", {})
                                store_info["original_pdf"] = metadata.get("original_pdf", "未知")
                                store_info["pdf_name"] = metadata.get("pdf_name", "未知")
                        except:
                            pass

                    return store_id, vector_store, store_info
                else:
                    logging.warning(f"  向量库 {store_id} 存在但文档数为0")
                    return None

            except Exception as e:
                logging.error(f"  加载向量库 {store_id} 失败: {e}")
                return None

        except Exception as e:
            logging.error(f"  处理向量库 {store_path} 时出错: {e}")
            return None

    def search_across_all_stores(
        self,
        query: str,
        k_per_store: Optional[int] = None,
        total_max_k: Optional[int] = None
    ) -> List[Any]:
        """在所有向量库中搜索"""
        all_results = []
        raw_k_per_store = k_per_store if k_per_store is not None else self.config.retrieval.k_per_store
        raw_total_max_k = total_max_k if total_max_k is not None else self.config.retrieval.total_max_k
        effective_k_per_store = max(1, int(raw_k_per_store or 3))
        effective_total_max_k = max(effective_k_per_store, int(raw_total_max_k or effective_k_per_store))

        for store_id, vector_store in self.vector_stores.items():
            try:
                # 执行相似度搜索
                results = vector_store.similarity_search_with_score(
                    query,
                    k=effective_k_per_store
                )

                # 添加来源信息
                for doc, score in results:
                    # 添加分数和来源信息
                    doc.metadata['similarity_score'] = float(score)
                    doc.metadata['source_store'] = store_id

                    # 确保必要元数据存在
                    if 'data_id' not in doc.metadata:
                        if store_id.startswith("kb_"):
                            doc.metadata['data_id'] = store_id[3:]
                        else:
                            doc.metadata['data_id'] = store_id

                    all_results.append(doc)

                if self.config.system.show_retrieval_info:
                    logging.debug(f"  {store_id}: 找到 {len(results)} 个文档")

            except Exception as e:
                logging.error(f"在向量库 {store_id} 中搜索失败: {e}")
                continue

        # 按相似度分数排序
        all_results.sort(key=lambda x: x.metadata.get('similarity_score', 0), reverse=True)

        # 去重（基于内容哈希）
        seen_contents = set()
        unique_results = []

        for doc in all_results:
            content_hash = hash(doc.page_content[:200])  # 基于前200个字符去重
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(doc)

        # 应用相似度阈值
        if self.config.retrieval.similarity_threshold > 0:
            filtered_results = [
                doc for doc in unique_results
                if doc.metadata.get('similarity_score', 0) >= self.config.retrieval.similarity_threshold
            ]
        else:
            filtered_results = unique_results

        # 应用重排序（如果启用）
        if self.reranker and self.config.retrieval.enable_reranking:
            reranked_results = self.reranker.rerank(query, filtered_results)
            if self.config.system.show_retrieval_info:
                logging.info(f"🔄 重排序完成: {len(filtered_results)} -> {len(reranked_results)} 个文档")
            final_results = reranked_results
        else:
            final_results = filtered_results

        # 限制总数量
        final_results = final_results[:effective_total_max_k]

        if self.config.system.show_retrieval_info:
            logging.info(f"🔍 跨库搜索完成: 从 {len(self.vector_stores)} 个库中找到 {len(final_results)} 个相关文档")

        return final_results

    def get_store_info(self) -> Dict[str, Dict]:
        """获取向量库信息"""
        return self.store_info

    def get_all_stores(self) -> Dict[str, Any]:
        """获取所有向量库"""
        return self.vector_stores


# ==================== 智能问答系统 ====================

class QASystem:
    """智能问答系统 - 兼容版"""

    def __init__(self, config: Optional[QASystemConfig] = None):
        self.config = config or UnifiedConfigManager.load_config()
        self.vector_manager = None
        self.llm = None
        self.qa_chain = None

        if not self._init_system():
            raise RuntimeError("系统初始化失败")

    def _init_system(self) -> bool:
        """初始化系统"""
        print("\n" + "=" * 60)
        print("🚀 智能问答系统初始化")
        print("=" * 60)

        # 1. 初始化向量库管理器
        print("1. 初始化向量库管理器...")
        try:
            self.vector_manager = VectorStoreManager(self.config)
            if not self.vector_manager.vector_stores:
                print("❌ 未成功加载任何向量库")
                print("   请确保已经:")
                print("   1) 运行第一阶段: python data_loader.py <文件或目录>")
                print("   2) 运行第二阶段: python vector_store.py --process-all")
                return False

            store_count = len(self.vector_manager.vector_stores)
            total_docs = sum(info.get('num_docs', 0) for info in self.vector_manager.store_info.values())
            print(f"✅ 加载 {store_count} 个向量库，共 {total_docs} 个文档")

        except Exception as e:
            print(f"❌ 向量库管理器初始化失败: {e}")
            return False

        # 2. 初始化LLM
        print("\n2. 初始化大语言模型...")
        try:
            if not self._init_llm():
                return False
            print(f"✅ LLM初始化完成: {self.config.llm.model_name}")
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            return False

        # 3. 创建问答链
        print("\n3. 创建问答链...")
        try:
            if not self._init_qa_chain():
                return False
            print("✅ 问答链创建完成")
        except Exception as e:
            print(f"❌ 问答链创建失败: {e}")
            return False

        print("\n" + "=" * 60)
        print("🎉 系统初始化成功!")
        print("=" * 60)

        # 显示系统状态
        self._show_system_status()

        return True

    def _init_llm(self) -> bool:
        """初始化大语言模型"""
        try:
            provider = self.config.llm.provider or "local"

            if provider == "local":
                return self._init_local_llm()
            elif provider == "api":
                return self._init_api_llm()
            else:
                print(f"❌ 未知的provider: {provider}")
                return False

        except Exception as e:
            print(f"❌ LLM初始化错误: {e}")
            return False

    def _init_local_llm(self) -> bool:
        """初始化本地LLM"""
        try:
            from langchain_ollama import ChatOllama

            # 检查Ollama服务
            import subprocess
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if self.config.llm.model_name not in result.stdout:
                    print(f"⚠️  模型 {self.config.llm.model_name} 未找到")
                    print(f"   请运行: ollama pull {self.config.llm.model_name}")

            except Exception as e:
                print(f"⚠️  Ollama检查失败: {e}")
                print(f"   确保Ollama服务已启动: ollama serve")

            # 创建LLM
            self.llm = ChatOllama(
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                num_predict=self.config.llm.num_predict,
                top_p=self.config.llm.top_p,
                top_k=self.config.llm.top_k,
                num_ctx=self.config.llm.num_ctx
            )

            print(f"✅ 本地LLM初始化完成: {self.config.llm.model_name}")
            return True

        except Exception as e:
            print(f"❌ 本地LLM初始化错误: {e}")
            return False

    def _init_api_llm(self) -> bool:
        """初始化API LLM"""
        try:
            from langchain_openai import ChatOpenAI

            # 检查API密钥
            api_key = self.config.llm.api_key
            if not api_key:
                print("❌ 请在config.yaml中设置llm.api_key")
                return False

            # 确定API基础URL
            api_base = self.config.llm.api_base
            if not api_base:
                # 默认使用DeepSeek API
                api_base = "https://api.deepseek.com"

            print(f"使用API: {api_base}")
            print(f"模型: {self.config.llm.model_name}")

            # 创建LLM
            self.llm = ChatOpenAI(
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.api_base,
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.num_predict,
                top_p=self.config.llm.top_p,
            )

            print(f"✅ API LLM初始化完成")
            return True

        except Exception as e:
            print(f"❌ API LLM初始化错误: {e}")
            return False

    def _init_qa_chain(self) -> bool:
        """创建问答链"""
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableLambda
            from langchain_core.output_parsers import StrOutputParser

            # 检索器函数
            def retrieve_documents(query_input):
                """检索文档"""
                retrieval_limits = {}
                if isinstance(query_input, dict):
                    cached_docs = query_input.get("retrieved_docs")
                    if cached_docs is not None:
                        return cached_docs
                    query = query_input.get("question", "")
                    retrieval_limits = query_input.get("retrieval_limits", {}) or {}
                else:
                    query = query_input

                if self.config.system.show_retrieval_info:
                    print(f"🔍 检索中...")

                docs = self.vector_manager.search_across_all_stores(
                    query,
                    k_per_store=retrieval_limits.get("k_per_store"),
                    total_max_k=retrieval_limits.get("total_max_k")
                )

                if self.config.system.show_retrieval_info:
                    print(f"   找到 {len(docs)} 个相关片段")

                return docs

            # 格式化文档函数
            def format_documents(docs):
                """格式化检索到的文档"""
                if not docs:
                    return "无相关上下文信息。"

                formatted = []
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content

                    # 获取元数据
                    metadata = doc.metadata
                    source = metadata.get('source', '未知文档')
                    page = metadata.get('page', '未知')
                    store = metadata.get('source_store', '未知库')
                    score = metadata.get('similarity_score', 0)

                    # 构建格式化字符串
                    formatted.append(
                        f"【片段 {i} | 相关度: {score:.3f}】\n"
                        f"来源: {store} - {source} (第{page}页)\n"
                        f"内容: {content}\n"
                    )

                return "\n" + "\n".join(formatted) + "\n"

            # 创建提示模板
            system_template = self.config.prompt.system_template
            human_template = self.config.prompt.human_template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                ("human", human_template)
            ])

            def extract_question(query_input):
                if isinstance(query_input, dict):
                    return query_input.get("question", "")
                return query_input

            # 构建链
            self.qa_chain = (
                    {
                        "context": RunnableLambda(retrieve_documents) | RunnableLambda(format_documents),
                        "question": RunnableLambda(extract_question)
                    }
                    | prompt
                    | self.llm
                    | StrOutputParser()
            )
            return True

        except Exception as e:
            print(f"❌ 创建问答链失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _show_system_status(self):
        """显示系统状态"""
        store_info = self.vector_manager.get_store_info()

        print("\n📊 系统状态:")
        print("-" * 50)
        print(f"📚 向量库数量: {len(store_info)}")

        total_docs = 0
        for store_id, info in store_info.items():
            docs = info.get('num_docs', 0)
            total_docs += docs

            # 显示简短信息
            pdf_name = info.get('pdf_name', '未知PDF')
            if len(pdf_name) > 20:
                pdf_name = pdf_name[:17] + "..."

            print(f"  - {store_id[:15]:15s} | {pdf_name:20s} | {docs:4d} 个文档")

        print("-" * 50)
        print(f"📄 总文档数: {total_docs}")
        print(f"🤖 模型: {self.config.llm.model_name}")
        print(f"🔍 检索配置: {self.config.retrieval.k_per_store} 个/库，最多 {self.config.retrieval.total_max_k} 个")
        print(f"📝 提示词: {len(self.config.prompt.system_template)} 字符")
        print("-" * 50)

    def _format_retrieved_docs(self, docs: List[Any]) -> List[Dict[str, Any]]:
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    def _get_dynamic_retrieval_limits(self, question: str) -> Dict[str, Any]:
        default_k_per_store = max(1, int(self.config.retrieval.k_per_store or 3))
        default_total_max_k = max(default_k_per_store, int(self.config.retrieval.total_max_k or default_k_per_store))
        dynamic_complexity = self.config.retrieval.dynamic_complexity or {}
        dynamic_enabled = bool(dynamic_complexity.get("enabled", True))
        simple_ratio = float(dynamic_complexity.get("simple_ratio", 0.5))
        medium_ratio = float(dynamic_complexity.get("medium_ratio", 1.0))
        complex_ratio = float(dynamic_complexity.get("complex_ratio", 1.5))
        min_k_per_store = max(1, int(dynamic_complexity.get("min_k_per_store", 1)))
        min_total_max_k = max(min_k_per_store, int(dynamic_complexity.get("min_total_max_k", min_k_per_store)))
        hard_cap_total_max_k = max(
            min_total_max_k,
            int(dynamic_complexity.get("hard_cap_total_max_k", default_total_max_k))
        )

        if not dynamic_enabled:
            return {
                "complexity": "medium",
                "k_per_store": default_k_per_store,
                "total_max_k": min(default_total_max_k, hard_cap_total_max_k)
            }

        prompt = (
            "你是检索复杂度分类器。请保守地判断问题复杂度，优先选择较简单的分类。"
                "仅输出一个标签：simple 或 medium 或 complex 或 no_retrieval。"
                "分类标准："
                "- simple：简单问候、简短问题、名词定义（如'你好'、'介绍你自己'、'Transformer的定义'）"
                "- medium：一般技术问题、操作指南、对比分析（如'介绍一下BERT'、'机器学习的应用'）"
                "- complex：需要深度推理、总结归纳、复杂逻辑的问题（如'总结《大模型基础》第二章的脉络'、'如何结合多种技术优化系统性能'）"
                "- no_retrieval：无需检索知识库，通用问题、闲聊、系统命令（如'?'、'help'、'退出'）"
                "判断原则：不确定时选择medium。"
            f"\n问题：{question}"
        )

        complexity = "medium"
        try:
            llm_result = self.llm.invoke(prompt)
            content = getattr(llm_result, "content", "")
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            content = str(content).lower()
            if "no_retrieval" in content:
                complexity = "no_retrieval"
            elif "complex" in content:
                complexity = "complex"
            elif "simple" in content:
                complexity = "simple"
            elif "medium" in content:
                complexity = "medium"
        except Exception:
            complexity = "medium"

        # 从配置中获取动态复杂度参数
        dynamic_config = getattr(self.config.retrieval, "dynamic_complexity", None)
        if not dynamic_config or not getattr(dynamic_config, "enabled", True):
            # 如果未启用动态复杂度，使用固定逻辑
            if complexity == "no_retrieval":
                # 无需检索：直接返回空结果，不进行向量检索
                k_per_store = 0
                total_max_k = 0
            elif complexity == "simple":
                # 简单查询：轻微减少检索量，保证基础信息覆盖
                k_per_store = max(3, int(round(default_k_per_store * 0.6)))
                total_max_k = max(k_per_store, int(round(default_total_max_k * 0.8)))
            elif complexity == "complex":
                # 复杂查询：适度增加检索量，提供更全面的信息
                k_per_store = max(default_k_per_store, int(round(default_k_per_store * 1.5)))
                total_max_k = max(k_per_store, min(default_total_max_k * 2, int(round(default_total_max_k * 1.2))))
            else:
                # 中等复杂度查询：保持默认检索量
                k_per_store = default_k_per_store
                total_max_k = default_total_max_k
        else:
            # 使用配置中的参数
            simple_ratio = getattr(dynamic_config, "simple_ratio", 0.5)
            medium_ratio = getattr(dynamic_config, "medium_ratio", 1.0)
            complex_ratio = getattr(dynamic_config, "complex_ratio", 1.5)
            no_retrieval_ratio = getattr(dynamic_config, "no_retrieval_ratio", 0.0)
            hard_cap_total_max_k = getattr(dynamic_config, "hard_cap_total_max_k", 20)

            ratio_map = {
                "simple": simple_ratio,
                "medium": medium_ratio,
                "complex": complex_ratio,
                "no_retrieval": no_retrieval_ratio
            }
            ratio = ratio_map.get(complexity, medium_ratio)
            
            if complexity == "no_retrieval":
                # 无需检索：直接返回空结果，不进行向量检索
                k_per_store = 0
                total_max_k = 0
            else:
                k_per_store = max(1, int(round(default_k_per_store * ratio)))
                total_max_k = max(k_per_store, int(round(default_total_max_k * ratio)))
                total_max_k = min(total_max_k, hard_cap_total_max_k)
                k_per_store = min(k_per_store, total_max_k)

        return {
            "complexity": complexity,
            "k_per_store": k_per_store,
            "total_max_k": total_max_k
        }

    def query(self, question: str, verbose: bool = True, return_retrieved_docs: bool = False) -> str | Dict[str, Any]:
        """
        查询知识库
        """
        if not self.qa_chain:
            error_msg = "系统未正确初始化，请检查初始化日志。"
            if return_retrieved_docs:
                return {"answer": error_msg, "retrieved_docs": []}
            return error_msg

        try:
            if verbose:
                print(f"\n❓ 问题: {question}")
                print("-" * 50)

            retrieval_limits = self._get_dynamic_retrieval_limits(question)
            retrieved_docs = self.vector_manager.search_across_all_stores(
                question,
                k_per_store=retrieval_limits["k_per_store"],
                total_max_k=retrieval_limits["total_max_k"]
            )
            payload = {
                "question": question,
                "retrieved_docs": retrieved_docs,
                "retrieval_limits": retrieval_limits
            }
            formatted_docs = self._format_retrieved_docs(retrieved_docs) if return_retrieved_docs else []

            # 执行查询
            if self.config.system.streaming_output:
                # 流式输出版本
                if verbose:
                    print("💡 答案: ", end="", flush=True)
                full_answer = ""

                for chunk in self.qa_chain.stream(payload):
                    full_answer += chunk
                    # 逐个字符输出，模拟打字效果
                    if verbose:
                        for char in chunk:
                            print(char, end="", flush=True)
                            if self.config.system.streaming_delay > 0:
                                import time
                                time.sleep(self.config.system.streaming_delay)

                if verbose:
                    print()  # 输出换行

                if return_retrieved_docs:
                    return {"answer": full_answer, "retrieved_docs": formatted_docs}
                return full_answer
            else:
                # 原有非流式输出
                answer = self.qa_chain.invoke(payload)

                if verbose:
                    print(f"💡 答案 ({len(answer)} 字符):")
                    print("-" * 50)
                    print(answer)

                if return_retrieved_docs:
                    return {"answer": answer, "retrieved_docs": formatted_docs}
                return answer

        except Exception as e:
            error_msg = f"查询过程中出现错误: {str(e)}"
            logging.error(error_msg)
            if return_retrieved_docs:
                return {"answer": error_msg, "retrieved_docs": []}
            return error_msg

    def stream_query(self, question: str):
        """流式查询（生成器版本）"""
        if not self.qa_chain:
            yield "系统未正确初始化，请检查初始化日志。"
            return

        try:
            # 使用流式输出
            full_answer = ""
            retrieval_limits = self._get_dynamic_retrieval_limits(question)
            payload = {"question": question, "retrieval_limits": retrieval_limits}
            for chunk in self.qa_chain.stream(payload):
                full_answer += chunk
                yield chunk
        except Exception as e:
            yield f"查询过程中出现错误: {str(e)}"

    def stream_query_with_docs(self, question: str):
        """流式查询并返回检索文档"""
        if not self.qa_chain:
            return iter(["系统未正确初始化，请检查初始化日志。"]), []

        retrieval_limits = self._get_dynamic_retrieval_limits(question)
        retrieved_docs = self.vector_manager.search_across_all_stores(
            question,
            k_per_store=retrieval_limits["k_per_store"],
            total_max_k=retrieval_limits["total_max_k"]
        )
        payload = {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "retrieval_limits": retrieval_limits
        }
        formatted_docs = self._format_retrieved_docs(retrieved_docs)

        def _stream():
            try:
                for chunk in self.qa_chain.stream(payload):
                    yield chunk
            except Exception as e:
                yield f"查询过程中出现错误: {str(e)}"

        return _stream(), formatted_docs
