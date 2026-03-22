#!/usr/bin/env python3
"""
向量存储管理
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Tuple, Generator, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
import time
import yaml
from tqdm import tqdm
from langchain_core.documents import Document
from dacite import from_dict


# ==================== 配置管理 ====================

@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    local_path: Optional[str] = None
    online_fallback: Optional[str] = None
    device: Optional[str] = None
    normalize_embeddings: Optional[bool] = None 
    model_kwargs: Optional[Dict[str, Any]] = None
    encode_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class BatchConfig:
    """批处理配置"""
    enabled: Optional[bool] = None
    batch_size: Optional[int] = None
    show_progress: Optional[bool] = None
    max_concurrent_batches: Optional[int] = None


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    persist_directory: Optional[str] = None
    collection_prefix: Optional[str] = None
    embedding: Optional[EmbeddingConfig] = None
    batch_processing: Optional[BatchConfig] = None


class ConfigManager:
    """配置管理器"""
    @staticmethod
    def load_vector_config(config_path: str = "config.yaml") -> VectorStoreConfig:
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        vector_config = config_dict.get('vector_processing', {})

        return from_dict(VectorStoreConfig, vector_config)


# ==================== 日志设置 ====================

def setup_logging(log_level: str = "INFO", log_file: str = "vector_store.log"):
    """设置日志"""
    # 创建日志目录
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)


# ==================== 核心向量存储管理器 ====================

class VectorStoreConnectionPool:
    """简单连接池"""
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.pools: Dict[str, dict] = {}
        self.lock = threading.RLock()

    def get_store(self, collection_name: str, embedding_model, persist_directory: str):
        with self.lock:
            # 如果已在池中，直接返回
            if collection_name in self.pools:
                self.pools[collection_name]["last_used"] = time.time()
                return self.pools[collection_name]["store"]

            # 如果池已满，淘汰最旧的
            if len(self.pools) >= self.max_size:
                self._evict_oldest()

            # 创建新连接
            from langchain_community.vectorstores import Chroma
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model,
                collection_name=collection_name
            )

            # 添加到池
            self.pools[collection_name] = {
                "store": vector_store,
                "last_used": time.time(),
                "created_at": time.time()
            }

            logger.debug(f"创建新向量库连接: {collection_name}")
            return vector_store

    def _evict_oldest(self):
        """淘汰最久未使用的连接"""
        if not self.pools:
            return

        oldest_name = min(self.pools.items(),
                          key=lambda x: x[1]["last_used"])[0]
        del self.pools[oldest_name]
        logger.debug(f"淘汰旧连接: {oldest_name}")


class VectorStoreManager:
    """向量存储管理器 - 支持分批处理"""
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """初始化向量存储管理器"""
        self.config = config or ConfigManager.load_vector_config()
        self.embedding_model = None
        self.vector_store = None

        # 创建向量库目录
        os.makedirs(self.config.persist_directory, exist_ok=True)
        logger.info(f"向量库目录: {os.path.abspath(self.config.persist_directory)}")

        # 初始化嵌入模型
        self._init_embedding_model()

        # 初始化连接池
        self.connection_pool = VectorStoreConnectionPool(max_size=5)
        logger.info("✅ 向量库连接池已初始化")

    def _init_embedding_model(self):
        """初始化嵌入模型"""
        from langchain_huggingface import HuggingFaceEmbeddings

        # 优先使用本地模型
        model_path = self.config.embedding.local_path

        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.warning(f"本地模型路径不存在: {model_path}")
            logger.info(f"使用在线模型: {self.config.embedding.online_fallback}")
            model_path = self.config.embedding.online_fallback

        try:
            # 设置模型参数
            model_kwargs = {'device': self.config.embedding.device}
            if self.config.embedding.model_kwargs:
                model_kwargs.update(self.config.embedding.model_kwargs)

            encode_kwargs = {}
            if self.config.embedding.normalize_embeddings:
                encode_kwargs['normalize_embeddings'] = True
            if self.config.embedding.encode_kwargs:
                encode_kwargs.update(self.config.embedding.encode_kwargs)

            # 创建嵌入模型
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logger.info(f"✅ 嵌入模型初始化完成: {os.path.basename(model_path)}")

        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise

    def _batch_documents(self, documents: List, batch_size: int) -> Generator[List, None, None]:
        """将文档列表分批"""
        for i in range(0, len(documents), batch_size):
            yield documents[i:i + batch_size]

    def create_or_load_vector_store(self, documents: List, data_id: str,
                                    force_recreate: bool = False) -> Tuple[Any, str, bool]:
        """创建或加载向量库"""
        # 1. 构造集合名称和存储路径
        collection_name = f"{self.config.collection_prefix}{data_id}"
        store_path = os.path.join(self.config.persist_directory, collection_name)

        # 2. 判断向量库是否已存在，如果已存在且不强制重建，则直接加载
        dir_exists = os.path.exists(store_path) and any(os.scandir(store_path))
        if not force_recreate and dir_exists:
            logger.info(f"📂 加载现有向量库: {collection_name}")
            try:
                from langchain_community.vectorstores import Chroma
                vector_store = Chroma(
                    persist_directory=store_path,
                    embedding_function=self.embedding_model,
                    collection_name=collection_name
                )
                logger.info(f"✅ 向量库加载成功: {collection_name}")
                return vector_store, store_path, True
            except Exception as e:
                logger.error(f"加载现有向量库失败，将重新创建: {e}")

        # 3. 创建新向量库
        logger.info(f"🆕 创建新向量库: {collection_name}")
        os.makedirs(store_path, exist_ok=True)  # 确保目录存在

        if self.config.batch_processing.enabled:
            vector_store = self._create_vector_store_with_batching(
                documents=documents,
                collection_name=collection_name,
                store_path=store_path
            )
        else:
            from langchain_community.vectorstores import Chroma
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=store_path,
                collection_name=collection_name
            )
            vector_store.persist()  # 确保持久化

        logger.info(f"✅ 向量库创建成功: {collection_name} ({len(documents)}个文档)")
        return vector_store, store_path, False

    def _create_vector_store_with_batching(self, documents: List, collection_name: str, store_path: str) -> Any:
        """使用分批处理创建向量库"""
        from langchain_community.vectorstores import Chroma

        batch_size = self.config.batch_processing.batch_size
        total_batches = (len(documents) - 1) // batch_size + 1

        logger.info(f"使用分批处理，批次大小: {batch_size}, 总批次数: {total_batches}")

        vector_store = None
        processed_count = 0

        # 使用进度条（如果启用）
        if self.config.batch_processing.show_progress:
            batch_iter = tqdm(self._batch_documents(documents, batch_size),
                              total=total_batches, desc="向量化处理")
        else:
            batch_iter = self._batch_documents(documents, batch_size)

        for batch_num, batch in enumerate(batch_iter, 1):
            batch_size_actual = len(batch)
            processed_count += batch_size_actual

            if not self.config.batch_processing.show_progress:
                logger.info(f"  处理批次 {batch_num}/{total_batches}: "
                            f"文档 {processed_count - batch_size_actual + 1}-{processed_count}")

            if batch_num == 1:
                # 第一批：创建向量库
                vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embedding_model,
                    persist_directory=store_path,
                    collection_name=collection_name
                )
            else:
                # 后续批次：添加到现有库
                vector_store.add_documents(batch)

        # 持久化
        if vector_store:
            vector_store.persist()

        return vector_store

    def get_vector_store(self, collection_name: str):
        """获取特定向量库"""
        from langchain_community.vectorstores import Chroma

        store_path = os.path.join(self.config.persist_directory, collection_name)
        
        if not os.path.exists(store_path):
            logger.error(f"向量库不存在: {collection_name}")
            return None

        try:
            vector_store = self.connection_pool.get_store(
                collection_name=collection_name,
                embedding_model=self.embedding_model,
                persist_directory=store_path
            )
            logger.info(f"✅ 获取向量库成功: {collection_name}")
            return vector_store

        except Exception as e:
            logger.error(f"获取向量库失败: {e}")
            return None


# ==================== 数据处理辅助类 ====================

class DataLoader:
    @staticmethod
    def load_file(data_file: str) -> List[Document]:
        """加载单个数据文件"""
        from langchain_core.documents import Document

        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for item in data:
            # 确保元数据包含必要字段
            metadata = item.get("metadata", {})
            if "data_id" not in metadata:
                # 从文件名推断data_id
                metadata["data_id"] = os.path.splitext(os.path.basename(data_file))[0]

            documents.append(
                Document(
                    page_content=item["page_content"],
                    metadata=metadata
                )
            )

        return documents


# ==================== 批处理管理器 ====================

class BatchProcessor:
    """批处理管理器 - 处理多个数据文件"""
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or ConfigManager.load_vector_config()
        self.vs_manager = VectorStoreManager(config)

    def process_single_file(self, data_file: str, data_id: Optional[str] = None,
                            force_recreate: bool = False) -> Dict:
        """处理单个数据文件，返回结果字典"""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        if data_id is None:
            data_id = os.path.splitext(os.path.basename(data_file))[0]

        logger.info(f"\n📄 处理文件: {os.path.basename(data_file)}")
        documents = DataLoader.load_file(data_file)  # 注意：之后会修改 DataLoader
        if not documents:
            raise ValueError(f"文件为空: {data_file}")

        vector_store, store_path, was_loaded = self.vs_manager.create_or_load_vector_store(
            documents=documents,
            data_id=data_id,
            force_recreate=force_recreate
        )

        result = {
            "data_file": data_file,
            "data_id": data_id,
            "collection_name": f"{self.config.collection_prefix}{data_id}",
            "store_path": store_path,
            "num_docs": len(documents),
            "was_loaded": was_loaded,
            "status": "success",
            "processing_time": datetime.now().isoformat()
        }

        action = "加载" if was_loaded else "创建"
        logger.info(f"✅ {action}成功: {data_id} ({len(documents)}个文档)")
        return result

    def process_data_directory(self, data_dir: str = "./processed_data",
                               force_recreate: bool = False) -> List[Dict]:
        """处理数据目录中的所有文件"""
        logger.info("=" * 60)
        logger.info(f"🚀 开始处理数据目录")
        logger.info(f"数据目录: {os.path.abspath(data_dir)}")
        logger.info(f"向量库目录: {os.path.abspath(self.config.persist_directory)}")
        logger.info(f"批处理模式: {'启用' if self.config.batch_processing.enabled else '禁用'}")
        logger.info("=" * 60)

        # 获取所有数据文件
        data_files = self._discover_data_files(data_dir)

        if not data_files:
            logger.error("❌ 没有找到数据文件")
            return []

        logger.info(f"找到 {len(data_files)} 个数据文件")

        results = []
        success_count = 0

        # 处理每个文件
        for data_file, data_id in tqdm(data_files, desc="处理数据文件", unit="file"):
            logger.info(f"\n📄 处理文件: {os.path.basename(data_file)}")

            try:
                # 加载数据
                documents = DataLoader.load_file(data_file)

                if not documents:
                    logger.warning(f"文件为空: {data_file}")
                    continue

                logger.info(f"加载 {len(documents)} 个文本块")

                # 创建/加载向量库
                vector_store, store_path, was_loaded = self.vs_manager.create_or_load_vector_store(
                    documents=documents,
                    data_id=data_id,
                    force_recreate=force_recreate
                )

                result = {
                    "data_file": data_file,
                    "data_id": data_id,
                    "collection_name": f"{self.config.collection_prefix}{data_id}",
                    "store_path": store_path,
                    "num_docs": len(documents),
                    "was_loaded": was_loaded,
                    "status": "success",
                    "processing_time": datetime.now().isoformat()
                }

                results.append(result)
                success_count += 1

                action = "加载" if was_loaded else "创建"
                logger.info(f"✅ {action}成功: {data_id} ({len(documents)}个文档)")

            except Exception as e:
                logger.error(f"❌ 处理失败 {data_file}: {e}")
                results.append({
                    "data_file": data_file,
                    "data_id": data_id,
                    "error": str(e),
                    "status": "failed",
                    "processing_time": datetime.now().isoformat()
                })

        # 生成报告
        if results:
            self._generate_processing_report(results, success_count, len(data_files))

        return results

    def _discover_data_files(self, data_dir: str) -> List[Tuple[str, str]]:
        """发现数据目录中的所有数据文件"""
        if not os.path.exists(data_dir):
            logger.error(f"数据目录不存在: {data_dir}")
            return []

        results = []

        # 方法1: 使用汇总文件
        summary_file = os.path.join(data_dir, "processing_summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)

                for file_info in summary.get("files", []):
                    data_file = file_info.get("data_file", "")
                    data_id = file_info.get("data_id", "")

                    # 处理可能的路径问题
                    if data_file and not os.path.exists(data_file):
                        # 尝试在数据目录中查找
                        alt_file = os.path.join(data_dir, f"{data_id}.json")
                        if os.path.exists(alt_file):
                            data_file = alt_file

                    if data_file and os.path.exists(data_file):
                        results.append((data_file, data_id))

                if results:
                    logger.info(f"从汇总文件发现 {len(results)} 个数据文件")
                    return results

            except Exception as e:
                logger.warning(f"汇总文件读取失败: {e}")

        # 方法2: 直接扫描目录
        for filename in os.listdir(data_dir):
            if filename.endswith('.json') and not filename.endswith(
                    '_meta.json') and filename != 'processing_summary.json':
                data_file = os.path.join(data_dir, filename)
                data_id = os.path.splitext(filename)[0]
                results.append((data_file, data_id))

        logger.info(f"扫描发现 {len(results)} 个数据文件")
        return results

    def _generate_processing_report(self, results: List[Dict], success_count: int, total_count: int):
        """生成处理报告（已弃用）"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 批处理完成!")
        logger.info("=" * 60)
        logger.info(f"✅ 成功处理: {success_count}/{total_count} 个文件")
        success_rate = f"{(success_count / total_count * 100):.1f}%" if total_count > 0 else "0%"
        logger.info(f"📊 成功率: {success_rate}")
        logger.info(f"📁 向量库目录: {os.path.abspath(self.config.persist_directory)}")


# ==================== 命令行接口 ====================

def main():
    """命令行主函数"""
    # 解析命令行参数（需先添加 --log-level 选项）
    log_level = "INFO"
    if "--log-level" in sys.argv:
        idx = sys.argv.index("--log-level")
        if idx + 1 < len(sys.argv):
            log_level = sys.argv[idx + 1]

    # 配置日志（仅在主程序执行时）
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("vector_store.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command in ["--help", "-h"]:
        print_usage()

    elif command == "--process-all":
        """处理所有数据文件"""
        force_recreate = "--force" in sys.argv

        # 获取数据目录（从命令行参数或配置）
        data_dir = "./processed_data"
        if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
            data_dir = sys.argv[2]

        processor = BatchProcessor()
        processor.process_data_directory(data_dir, force_recreate)


    elif command in ["--process", "-p"]:
        """处理单个数据文件"""
        if len(sys.argv) < 3:
            print("❌ 请提供数据文件路径")
            sys.exit(1)

        data_file = sys.argv[2]
        data_id = sys.argv[3] if len(sys.argv) > 3 else None
        force_recreate = "--force" in sys.argv

        # 检查文件是否存在
        if not os.path.exists(data_file):
            # 尝试在默认数据目录中查找
            default_dir = "./processed_data"
            if os.path.exists(default_dir):
                possible_path = os.path.join(default_dir, data_file)
                if os.path.exists(possible_path):
                    data_file = possible_path
                else:
                    # 尝试添加.json后缀
                    if not data_file.endswith('.json'):
                        possible_path = os.path.join(default_dir, data_file + '.json')
                        if os.path.exists(possible_path):
                            data_file = possible_path
            
            if not os.path.exists(data_file):
                print(f"❌ 数据文件不存在: {data_file}")
                sys.exit(1)

        processor = BatchProcessor()

        try:
            result = processor.process_single_file(data_file, data_id, force_recreate)
            print(f"\n🎉 处理成功!")
            print(f"📁 存储路径: {result['store_path']}")
            print(f"📊 文档数量: {result['num_docs']}")

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            sys.exit(1)
    else:
        print(f"❌ 未知命令: {command}")
        print_usage()


def print_usage():
    """打印使用说明"""
    print("=" * 60)
    print("📚 向量存储管理器")
    print("=" * 60)
    print("用法: python vector_store.py <命令> [参数]")
    print()
    print("命令:")
    print("  --process-all [数据目录]    处理所有数据文件")
    print("  --process, -p <文件>      处理单个数据文件")
    print("  --help, -h                 显示帮助信息")
    print()
    print("参数:")
    print("  --force                   强制重新创建向量库")
    print("  <数据目录>                指定数据目录（默认: ./processed_data）")
    print()
    print("示例:")
    print("  python vector_store.py --process-all")
    print("  python vector_store.py --process-all ./processed_data")
    print("  python vector_store.py --process data.json")
    print("=" * 60)


if __name__ == "__main__":
    main()